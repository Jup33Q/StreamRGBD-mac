#!/usr/bin/env python3
"""LoRA-enhanced CoreML img2img pipeline."""
import os
import sys
import time
import gc
import threading

import numpy as np
import cv2
import torch

# Python 3.12 removed distutils; coremltools still imports it.
# Shim via setuptools before importing coremltools.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "coreml_models")
LORAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "loras")

from configs import MODEL_CONFIGS, DEFAULT_PROMPTS  # noqa: E402
from pipelines.coreml import ensure_vae_encoder, ensure_vae_decoder


class LoRAEnhancedPipeline:
    """CoreML img2img pipeline with LoRA support."""

    def __init__(self, model_name, render_size, output_size, prompt,
                 strength=0.5, prompts=None, latent_feedback=0.3, coreml_dir=COREML_DIR,
                 lora_paths=None, lora_weights=None):
        import coremltools as ct

        self.model_name = model_name
        self.render_size = render_size
        self.output_size = output_size
        self.latent_size = render_size // 8
        cfg = MODEL_CONFIGS[model_name]
        lora_paths = lora_paths or []
        lora_weights = lora_weights or [0.8] * len(lora_paths)

        print("\n--- Loading CoreML Models ---")
        enc_path = ensure_vae_encoder(render_size, coreml_dir)
        dec_path = ensure_vae_decoder(render_size, coreml_dir)
        prefix = cfg["unet_prefix"]
        unet_path = os.path.join(coreml_dir, f"{prefix}.mlpackage")
        if not os.path.exists(unet_path):
            print(f"ERROR: UNet model not found at {unet_path}")
            print(f"  Run: python scripts/convert_models.py --model {model_name}")
            sys.exit(1)
        print(f"  UNet: {unet_path}")

        cu = ct.ComputeUnit.CPU_AND_GPU
        self.vae_encoder = ct.models.MLModel(enc_path, compute_units=cu)
        self.vae_decoder = ct.models.MLModel(dec_path, compute_units=cu)
        self.unet = ct.models.MLModel(unet_path, compute_units=cu)

        # Buffers
        self._img_buf = np.empty((1, 3, render_size, render_size), dtype=np.float16)
        self._lat_buf = np.empty((1, 4, self.latent_size, self.latent_size), dtype=np.float16)
        self._out_buf = np.empty((1, 4, self.latent_size, self.latent_size), dtype=np.float16)
        self._t_buf = np.empty((1,), dtype=np.float16)
        self._norm_lut = (np.arange(256, dtype=np.float32) / 127.5 - 1.0).astype(np.float16)

        # Load diffusers pipeline for text encoder + LoRA support
        print("  Loading text encoder + LoRA...")
        pipe = __import__('diffusers').StableDiffusionPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16).to("mps")

        # Apply LoRAs if provided
        if lora_paths:
            for lora_path, weight in zip(lora_paths, lora_weights):
                if not os.path.exists(lora_path):
                    print(f"  WARNING: LoRA not found: {lora_path}")
                    continue
                print(f"  Loading LoRA: {os.path.basename(lora_path)} (weight={weight})")
                try:
                    pipe.load_lora_weights(lora_path)
                    pipe.fuse_lora(lora_scale=weight)
                    print(f"  LoRA fused successfully")
                except Exception as e:
                    print(f"  WARNING: Failed to load LoRA {lora_path}: {e}")

        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder

        # Scheduler setup (same as original)
        if cfg["scheduler"] == "euler":
            from diffusers import EulerDiscreteScheduler
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(1 if cfg["scheduler"] == "euler" else 50, device="mps")

        if cfg["scheduler"] == "euler":
            actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[
                min(int(actual_t), len(pipe.scheduler.alphas_cumprod) - 1)
            ].item()
        else:
            t_idx = max(0, int(50 * (1.0 - strength)))
            if t_idx < len(pipe.scheduler.timesteps):
                actual_t = pipe.scheduler.timesteps[t_idx].cpu().item()
            else:
                actual_t = pipe.scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = pipe.scheduler.alphas_cumprod[int(actual_t)].item()

        self._sqrt_a = np.float16(np.sqrt(ap))
        self._sqrt_1ma = np.float16(np.sqrt(1.0 - ap))

        # Encode all prompts
        self._all_prompts = prompts if prompts else [prompt]
        self._prompt_index = 0
        self._all_embeds = []
        print(f"  Encoding {len(self._all_prompts)} prompt(s)...")
        for p in self._all_prompts:
            self._all_embeds.append(self._encode_single(p))
        self._prompt_embeds = self._all_embeds[0]
        self._current_prompt = self._all_prompts[0]
        self._target_embeds = self._prompt_embeds.copy()
        self._prompt_lerp_speed = 0.05

        del pipe
        gc.collect()
        torch.mps.empty_cache()

        # Fixed noise for temporal coherence
        rng = np.random.RandomState(42)
        self._fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(np.float16)
        self._prev_denoised = None
        self.latent_feedback = latent_feedback

        print("  Warming up CoreML...")
        self._warmup()
        print("  Ready!")

    def _encode_single(self, prompt):
        import torch
        with torch.no_grad():
            ti = self._tokenizer(
                prompt, padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            return self._text_encoder(ti.input_ids.to("mps"))[0].cpu().to(torch.float16).numpy()

    def next_prompt(self):
        self._prompt_index = (self._prompt_index + 1) % len(self._all_prompts)
        self._target_embeds = self._all_embeds[self._prompt_index]
        self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def prev_prompt(self):
        self._prompt_index = (self._prompt_index - 1) % len(self._all_prompts)
        self._target_embeds = self._all_embeds[self._prompt_index]
        self._current_prompt = self._all_prompts[self._prompt_index]
        return self._current_prompt

    def _warmup(self, n=25):
        np.copyto(self._img_buf, np.random.randn(1, 3, self.render_size, self.render_size).astype(np.float16))
        for _ in range(n):
            e = self.vae_encoder.predict({"image": self._img_buf})
            np.copyto(self._lat_buf, np.array(e["latent"]).astype(np.float16))
            u = self.unet.predict({
                "sample": self._lat_buf, "timestep": self._t_buf,
                "encoder_hidden_states": self._prompt_embeds,
            })
            np.copyto(self._out_buf, np.array(u["noise_pred"]).astype(np.float16))
            self.vae_decoder.predict({"latent": self._out_buf})

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if w > h:
            off = (w - h) // 2
            frame_bgr = frame_bgr[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            frame_bgr = frame_bgr[off:off + w, :]

        resized = cv2.resize(frame_bgr, (self.render_size, self.render_size))
        rgb = resized[:, :, ::-1]
        np.copyto(self._img_buf, self._norm_lut[rgb].transpose(2, 0, 1)[np.newaxis])

        # Smooth prompt transition
        diff = self._target_embeds - self._prompt_embeds
        if np.abs(diff).max() > 1e-4:
            self._prompt_embeds = self._prompt_embeds + self._prompt_lerp_speed * diff

        # VAE Encode
        enc = self.vae_encoder.predict({"image": self._img_buf})
        clean = np.array(enc["latent"]).astype(np.float16)

        # Latent feedback
        if self._prev_denoised is not None and self.latent_feedback > 0:
            fb = np.float16(self.latent_feedback)
            clean = (1.0 - fb) * clean + fb * self._prev_denoised

        # Add fixed noise
        noisy = self._sqrt_a * clean + self._sqrt_1ma * self._fixed_noise
        np.copyto(self._lat_buf, noisy)

        # UNet inference
        u = self.unet.predict({
            "sample": self._lat_buf, "timestep": self._t_buf,
            "encoder_hidden_states": self._prompt_embeds,
        })
        npred = np.array(u["noise_pred"]).astype(np.float16)
        denoised = (noisy - self._sqrt_1ma * npred) / self._sqrt_a

        self._prev_denoised = denoised.copy()
        np.copyto(self._out_buf, denoised)

        # VAE Decode
        dec = self.vae_decoder.predict({"latent": self._out_buf})
        r = np.array(dec["image"]).astype(np.float32).squeeze(0).transpose(1, 2, 0)
        r = ((r + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if r.shape[0] != self.output_size:
            r = cv2.resize(r, (self.output_size, self.output_size))
        return cv2.cvtColor(r, cv2.COLOR_RGB2BGR)


def resolve_lora_paths(lora_names, lora_dir):
    """Resolve LoRA names to full paths."""
    paths = []
    for name in lora_names:
        # Direct path check
        if os.path.exists(name):
            paths.append(name)
            continue
        # Check in lora_dir
        lora_path = os.path.join(lora_dir, f"{name}.safetensors")
        if os.path.exists(lora_path):
            paths.append(lora_path)
        else:
            print(f"WARNING: LoRA '{name}' not found in {lora_dir}")
    return paths


def list_local_loras(lora_dir):
    """List downloaded LoRAs."""
    if not os.path.exists(lora_dir):
        print(f"LoRA directory not found: {lora_dir}")
        print("Run: python download_loras.py --all")
        return
    files = sorted([f for f in os.listdir(lora_dir) if f.endswith('.safetensors')])
    if not files:
        print(f"No LoRAs in {lora_dir}")
        print("Run: python download_loras.py --all")
        return
    print(f"Local LoRAs ({len(files)}):")
    for f in files:
        size = os.path.getsize(os.path.join(lora_dir, f)) / (1024 * 1024)
        print(f"  {f:40s}  {size:6.1f} MB")
