#!/usr/bin/env python3
"""
StreamDiffusion for Mac — PyTorch/MPS Pipeline with Runtime LoRA Adapters

A PyTorch-based drop-in replacement for pipelines.coreml.Pipeline when LoRA
sliders need to affect the UNet at inference time.
"""
import os
import sys
import gc
import threading

import numpy as np
import cv2
import torch

from pipelines.coreml import COREML_DIR
from configs import MODEL_CONFIGS
from utils.device import _default_device
from depth.estimators import DepthEstimator


class TorchPipeline:
    """PyTorch img2img pipeline with temporal coherence and runtime LoRA adapters."""

    def __init__(self, model_name, render_size, output_size, prompt,
                 strength=0.5, prompts=None, latent_feedback=0.3, coreml_dir=COREML_DIR, seed=42,
                 lora_stack=None):
        self.model_name = model_name
        self.render_size = render_size
        self.output_size = output_size
        self.latent_size = render_size // 8
        self._seed = seed
        self._lora_stack = list(lora_stack or [])
        self._encode_lock = threading.Lock()
        self._loaded_lora_adapters = {}  # adapter_name -> path
        cfg = MODEL_CONFIGS[model_name]

        print("\n--- Loading PyTorch Pipeline (LoRA capable) ---")
        print(f"  Model: {cfg['model_id']}")
        self.device = _default_device()
        print(f"  Device: {self.device}")

        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

        self._pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Scheduler setup matching pipelines.coreml.Pipeline
        if cfg["scheduler"] == "euler":
            self._pipe.scheduler = EulerDiscreteScheduler.from_config(self._pipe.scheduler.config)
        self._pipe.scheduler.set_timesteps(1 if cfg["scheduler"] == "euler" else 50, device=self.device)

        if cfg["scheduler"] == "euler":
            actual_t = self._pipe.scheduler.timesteps[0].cpu().item()
            self._t = np.float16(actual_t)
            ap = self._pipe.scheduler.alphas_cumprod[
                min(int(actual_t), len(self._pipe.scheduler.alphas_cumprod) - 1)
            ].item()
        else:
            t_idx = max(0, int(50 * (1.0 - strength)))
            if t_idx < len(self._pipe.scheduler.timesteps):
                actual_t = self._pipe.scheduler.timesteps[t_idx].cpu().item()
            else:
                actual_t = self._pipe.scheduler.timesteps[0].cpu().item()
            self._t = np.float16(actual_t)
            ap = self._pipe.scheduler.alphas_cumprod[int(actual_t)].item()

        self._sqrt_a = np.float16(np.sqrt(ap))
        self._sqrt_1ma = np.float16(np.sqrt(1.0 - ap))
        self._t_tensor = torch.tensor([self._t], dtype=torch.float16, device=self.device)

        # Extract components for manual inference
        self.vae = self._pipe.vae
        self.unet = self._pipe.unet
        self._tokenizer = self._pipe.tokenizer
        self._text_encoder = self._pipe.text_encoder

        # VAE scaling factor
        self._vae_scale = getattr(self.vae.config, "scaling_factor", 0.18215)

        # Load initial LoRA adapters
        self._apply_lora_stack()

        # Encode prompts
        self._all_prompts = prompts if prompts else [prompt]
        self._prompt_index = 0
        self._all_embeds = []
        print(f"  Encoding {len(self._all_prompts)} prompt(s)...")
        for p in self._all_prompts:
            self._all_embeds.append(self._encode_single(p))
        with self._encode_lock:
            self._prompt_embeds = self._all_embeds[0]
            self._current_prompt = self._all_prompts[0]
            self._target_embeds = self._prompt_embeds.copy()
        self._prompt_lerp_speed = 0.05

        # Fixed noise for temporal coherence
        rng = np.random.RandomState(self._seed)
        self._fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(np.float16)
        self._prev_denoised = None
        self.latent_feedback = latent_feedback
        print(f"  Seed: {self._seed}")

        self._warmup()
        print("  Ready!")

    def _vae_encode(self, img):
        """Encode image to latent, handling both standard VAE and TinyVAE."""
        encoded = self.vae.encode(img)
        if hasattr(encoded, "latent_dist"):
            latent = encoded.latent_dist.sample()
        else:
            latent = encoded.latents
        return latent * self._vae_scale

    def _vae_decode(self, latent):
        """Decode latent to image, handling both standard VAE and TinyVAE."""
        decoded = self.vae.decode(latent / self._vae_scale)
        return decoded.sample

    def _encode_single(self, prompt):
        """Encode a single prompt into text embeddings."""
        with torch.no_grad():
            ti = self._tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeds = self._text_encoder(ti.input_ids.to(self.device))[0]
            return embeds.to(torch.float16).cpu().numpy()

    def _apply_lora_stack(self):
        """Load and activate LoRA adapters according to self._lora_stack."""
        active_loras = [l for l in self._lora_stack if l.get("weight", 0.0) != 0.0]

        # Determine which adapters we need loaded
        needed = {}
        for lora in active_loras:
            path = lora.get("path", "")
            if not path or not os.path.exists(path):
                print(f"    WARNING: LoRA not found: {path}")
                continue
            name = self._adapter_name(path)
            needed[name] = path

        # Unload adapters that are no longer needed to save memory
        current = set(self._loaded_lora_adapters.keys())
        to_remove = current - set(needed.keys())
        if to_remove:
            # diffusers 0.39: unload specific adapters is not directly exposed,
            # so we unload all and will reload the needed ones below.
            self._pipe.unload_lora_weights()
            self._loaded_lora_adapters = {}
            current = set()

        # Load any newly needed adapters
        for name, path in needed.items():
            if name not in self._loaded_lora_adapters:
                print(f"    Loading LoRA adapter: {name}")
                try:
                    self._pipe.load_lora_weights(path, adapter_name=name)
                    self._loaded_lora_adapters[name] = path
                except Exception as e:
                    print(f"    WARNING: Failed to load LoRA {name}: {e}")

        # Activate adapters with weights
        valid_adapters = []
        valid_weights = []
        for lora in active_loras:
            path = lora.get("path", "")
            name = self._adapter_name(path)
            if name in self._loaded_lora_adapters:
                valid_adapters.append(name)
                valid_weights.append(float(lora.get("weight", 0.0)))

        if valid_adapters:
            print(f"  Activating LoRA adapters: {list(zip(valid_adapters, valid_weights))}")
            self._pipe.set_adapters(valid_adapters, adapter_weights=valid_weights)
        else:
            # Deactivate all adapters
            try:
                self._pipe.set_adapters([])
            except Exception:
                pass

    @staticmethod
    def _adapter_name(path):
        """Create a safe adapter name from a LoRA file path."""
        base = os.path.basename(path)
        # Remove extension
        name = base.rsplit(".", 1)[0]
        # Replace characters that may confuse diffusers
        name = name.replace(".", "_").replace("-", "_")
        return name

    def _reencode_prompts(self):
        """Re-encode all prompts after LoRA / text encoder change."""
        print(f"  Re-encoding {len(self._all_prompts)} prompt(s) with updated LoRA stack...")
        self._all_embeds = [self._encode_single(p) for p in self._all_prompts]
        with self._encode_lock:
            self._prompt_embeds = self._all_embeds[self._prompt_index]
            self._target_embeds = self._prompt_embeds.copy()

    def set_lora_stack(self, lora_stack):
        """Replace the LoRA stack at runtime and re-encode prompts."""
        self._lora_stack = list(lora_stack or [])
        print(f"  Updating LoRA stack: {self._lora_stack}")
        self._apply_lora_stack()
        self._reencode_prompts()
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        print("  LoRA stack updated")
        return self._lora_stack

    def set_prompt(self, prompt):
        """Set a new prompt at runtime and re-encode it."""
        embeds = self._encode_single(prompt)
        self._all_prompts = [prompt]
        self._all_embeds = [embeds]
        self._prompt_index = 0
        with self._encode_lock:
            self._target_embeds = embeds
            self._prompt_embeds = embeds.copy()
        self._current_prompt = prompt
        return prompt

    def set_seed(self, seed):
        """Set a new random seed and regenerate fixed noise."""
        self._seed = int(seed)
        rng = np.random.RandomState(self._seed)
        self._fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(np.float16)
        self._prev_denoised = None
        print(f"  Seed updated: {self._seed}")
        return self._seed

    def next_prompt(self):
        self._prompt_index = (self._prompt_index + 1) % len(self._all_prompts)
        with self._encode_lock:
            self._target_embeds = self._all_embeds[self._prompt_index]
            self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def prev_prompt(self):
        self._prompt_index = (self._prompt_index - 1) % len(self._all_prompts)
        with self._encode_lock:
            self._target_embeds = self._all_embeds[self._prompt_index]
            self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def _warmup(self, n=2):
        """Run a few warmup iterations to allocate MPS memory."""
        dummy_img = torch.randn(
            1, 3, self.render_size, self.render_size,
            dtype=torch.float16, device=self.device
        )
        with torch.no_grad():
            for _ in range(n):
                latent = self._vae_encode(dummy_img)
                _ = self.unet(
                    latent,
                    self._t_tensor,
                    encoder_hidden_states=torch.from_numpy(self._prompt_embeds).to(self.device),
                ).sample
                _ = self._vae_decode(latent)
        if self.device == "mps":
            torch.mps.synchronize()

    def process_frame(self, frame_bgr):
        """Full pipeline accepting BGR frames and returning BGR."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = self.process_frame_rgb(rgb)
        return cv2.cvtColor(r, cv2.COLOR_RGB2BGR)

    def process_frame_rgb(self, frame_rgb):
        """Same pipeline but accepts and returns RGB frames."""
        import torch

        h, w = frame_rgb.shape[:2]
        if w > h:
            off = (w - h) // 2
            frame_rgb = frame_rgb[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            frame_rgb = frame_rgb[off:off + w, :]

        resized = cv2.resize(frame_rgb, (self.render_size, self.render_size))
        img = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device, dtype=torch.float16)
        img = img / 127.5 - 1.0

        # Smooth prompt transition (thread-safe)
        with self._encode_lock:
            diff = self._target_embeds - self._prompt_embeds
            if np.abs(diff).max() > 1e-4:
                self._prompt_embeds = self._prompt_embeds + self._prompt_lerp_speed * diff
            prompt_embeds = torch.from_numpy(self._prompt_embeds).to(self.device)

        # VAE Encode
        with torch.no_grad():
            latent = self._vae_encode(img)
            clean = latent.to(torch.float16)

        # Latent feedback from previous frame
        if self._prev_denoised is not None and self.latent_feedback > 0:
            fb = np.float16(self.latent_feedback)
            clean = (1.0 - fb) * clean + fb * torch.from_numpy(self._prev_denoised).to(self.device)

        # Add fixed noise
        noisy = (
            self._sqrt_a * clean
            + self._sqrt_1ma * torch.from_numpy(self._fixed_noise).to(self.device)
        )

        # UNet inference
        with torch.no_grad():
            noise_pred = self.unet(
                noisy,
                self._t_tensor,
                encoder_hidden_states=prompt_embeds,
            ).sample

        # Denoise
        denoised = (noisy - self._sqrt_1ma * noise_pred) / self._sqrt_a
        self._prev_denoised = denoised.to(torch.float16).cpu().numpy()

        # VAE Decode
        with torch.no_grad():
            decoded = self._vae_decode(denoised)

        # Postprocess
        r = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        r = ((r + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if r.shape[0] != self.output_size:
            r = cv2.resize(r, (self.output_size, self.output_size))
        return r


class TorchRGBDPipeline(TorchPipeline):
    """PyTorch pipeline extended with RGBD output."""

    def __init__(self, depth_model="auto", depth_backend="auto",
                 depth_coreml_path=None, output_width=None, output_height=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_width = output_width or self.output_size
        self.output_height = output_height or self.output_size
        print("\n--- Loading Depth Estimator ---")
        self.depth_estimator = DepthEstimator(
            model_name=depth_model,
            backend=depth_backend,
            coreml_path=depth_coreml_path,
        )

    def process_frame_rgbd(self, frame_bgr):
        """Run img2img + depth estimation and return RGB, depth, and RGBD."""
        from pipelines.rgbd import _crop_to_aspect_no_resize
        result_bgr = self.process_frame(frame_bgr)

        if self.output_width != self.output_height:
            cropped_bgr = _crop_to_aspect_no_resize(result_bgr, self.output_width, self.output_height)
            rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            depth_u8 = self.depth_estimator.estimate(rgb)
            result_bgr = cv2.resize(cropped_bgr, (self.output_width, self.output_height), interpolation=cv2.INTER_LANCZOS4)
            rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            depth_u8 = cv2.resize(depth_u8, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
        else:
            rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            depth_u8 = self.depth_estimator.estimate(rgb)

        rgbd = np.concatenate([rgb, depth_u8[:, :, np.newaxis]], axis=2)
        return result_bgr, depth_u8, rgbd
