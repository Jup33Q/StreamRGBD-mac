#!/usr/bin/env python3
"""
StreamDiffusion for Mac — CoreML img2img pipeline.

CoreML-accelerated real-time image-to-image transformation using
diffusion models on Apple Silicon.
"""
import os
import sys
import time
import gc
import threading

# Python 3.12 removed distutils; coremltools still imports it.
# Shim via setuptools before importing coremltools.
if "distutils" not in sys.modules:
    import setuptools
    sys.modules["distutils"] = setuptools._distutils

import numpy as np
import cv2
import coremltools as ct

COREML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "coreml_models")

# Central JSON-backed configuration.
from configs import MODEL_CONFIGS, DEFAULT_PROMPTS  # noqa: E402
from utils.hf_utils import from_pretrained_local_first  # noqa: E402


def ensure_vae_encoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Encoder if not present."""
    path = os.path.join(coreml_dir, f"taesd_encoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    print(f"  Auto-converting TinyVAE Encoder ({render_size}x{render_size})...")
    vae = from_pretrained_local_first(
        AutoencoderTiny.from_pretrained, "madebyollin/taesd"
    ).eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.encoder = v.encoder
        def forward(self, x):
            return self.encoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 3, render_size, render_size)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="latent", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    print(f"  Saved: {path}")
    return path


def ensure_vae_decoder(render_size, coreml_dir):
    """Auto-convert TinyVAE Decoder if not present."""
    if render_size == 512:
        path = os.path.join(coreml_dir, "taesd_decoder.mlpackage")
    else:
        path = os.path.join(coreml_dir, f"taesd_decoder_{render_size}.mlpackage")
    if os.path.exists(path):
        return path
    import torch
    from diffusers import AutoencoderTiny
    ls = render_size // 8
    print(f"  Auto-converting TinyVAE Decoder ({render_size}x{render_size})...")
    vae = from_pretrained_local_first(
        AutoencoderTiny.from_pretrained, "madebyollin/taesd"
    ).eval().float().cpu()

    class W(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.decoder = v.decoder
        def forward(self, x):
            return self.decoder(x)

    w = W(vae).eval()
    d = torch.randn(1, 4, ls, ls)
    with torch.no_grad():
        traced = torch.jit.trace(w, d)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=d.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="image", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    m.save(path)
    del m, traced, w, vae
    gc.collect()
    print(f"  Saved: {path}")
    return path


class Pipeline:
    """CoreML img2img pipeline with temporal coherence."""

    def __init__(self, model_name, render_size, output_size, prompt,
                 strength=0.5, prompts=None, latent_feedback=0.3, coreml_dir=COREML_DIR, seed=42,
                 lora_stack=None):
        import torch
        if coreml_dir is None:
            coreml_dir = COREML_DIR
        self.model_name = model_name
        self.render_size = render_size
        self.output_size = output_size
        self.latent_size = render_size // 8
        self._seed = seed
        self._lora_stack = list(lora_stack or [])
        self._encode_lock = threading.Lock()
        cfg = MODEL_CONFIGS[model_name]

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

        # Prompt encoding + LoRA
        self._pipe = None  # base diffusers pipeline (kept while loading LoRAs)
        self._load_text_encoder_with_loras()

        # Scheduler setup
        if cfg["scheduler"] == "euler":
            from diffusers import EulerDiscreteScheduler
            self._scheduler = EulerDiscreteScheduler.from_config(self._scheduler.config)
        self._scheduler.set_timesteps(1 if cfg["scheduler"] == "euler" else 50, device="mps")

        if cfg["scheduler"] == "euler":
            actual_t = self._scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = self._scheduler.alphas_cumprod[
                min(int(actual_t), len(self._scheduler.alphas_cumprod) - 1)
            ].item()
        else:
            t_idx = max(0, int(50 * (1.0 - strength)))
            if t_idx < len(self._scheduler.timesteps):
                actual_t = self._scheduler.timesteps[t_idx].cpu().item()
            else:
                actual_t = self._scheduler.timesteps[0].cpu().item()
            self._t_buf[0] = np.float16(actual_t)
            ap = self._scheduler.alphas_cumprod[int(actual_t)].item()

        self._sqrt_a = np.float16(np.sqrt(ap))
        self._sqrt_1ma = np.float16(np.sqrt(1.0 - ap))

        # Encode all prompts
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

        # Release the full diffusers pipeline if it was kept for LoRA fusion.
        self._pipe = None
        gc.collect()
        torch.mps.empty_cache()

        # Fixed noise for temporal coherence
        rng = np.random.RandomState(self._seed)
        self._fixed_noise = rng.randn(1, 4, self.latent_size, self.latent_size).astype(np.float16)
        self._prev_denoised = None
        self.latent_feedback = latent_feedback
        print(f"  Seed: {self._seed}")

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

    def set_prompt(self, prompt):
        """Set a new prompt at runtime and re-encode it."""
        import torch
        with torch.no_grad():
            ti = self._tokenizer(
                prompt,
                padding="max_length",
                max_length=self._tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            embeds = (
                self._text_encoder(ti.input_ids.to("mps"))[0]
                .cpu()
                .to(torch.float16)
                .numpy()
            )
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

    @staticmethod
    def _load_scheduler(model_id):
        """Load the repo's scheduler from its config alone (no full pipeline)."""
        import json
        import diffusers
        from huggingface_hub import hf_hub_download

        cfg_path = from_pretrained_local_first(
            hf_hub_download, model_id, filename="scheduler/scheduler_config.json"
        )
        with open(cfg_path) as f:
            sched_cfg = json.load(f)
        sched_cls = getattr(diffusers, sched_cfg.get("_class_name", "PNDMScheduler"))
        return sched_cls.from_config(sched_cfg)

    def _load_text_encoder_with_loras(self):
        """Load tokenizer + text encoder (+ scheduler), fusing active LoRAs.

        Without active LoRAs only the tokenizer, text encoder and scheduler are
        loaded, which is much faster than the full StableDiffusionPipeline and
        also tolerates partially-cached HF snapshots (single-file resolution
        instead of a whole-snapshot download). With active LoRAs the full
        pipeline is required for load_lora_weights/fuse_lora.
        """
        import torch
        cfg = MODEL_CONFIGS[self.model_name]
        model_id = cfg["model_id"]

        active_loras = [l for l in self._lora_stack if l.get("weight", 0.0) != 0.0]
        if not active_loras:
            print("  Loading text encoder...")
            from transformers import CLIPTextModel, CLIPTokenizer
            self._tokenizer = from_pretrained_local_first(
                CLIPTokenizer.from_pretrained, model_id, subfolder="tokenizer"
            )
            self._text_encoder = from_pretrained_local_first(
                CLIPTextModel.from_pretrained, model_id,
                subfolder="text_encoder", torch_dtype=torch.float16,
            ).to("mps")
            self._scheduler = self._load_scheduler(model_id)
            self._pipe = None
            return

        print(f"  Loading text encoder (full pipeline, applying {len(active_loras)} LoRA(s))...")
        pipe = from_pretrained_local_first(
            __import__('diffusers').StableDiffusionPipeline.from_pretrained,
            model_id,
            torch_dtype=torch.float16,
        ).to("mps")

        for lora in active_loras:
            path = lora.get("path", "")
            weight = float(lora.get("weight", 0.0))
            if not path or not os.path.exists(path):
                print(f"    WARNING: LoRA not found or disabled: {path}")
                continue
            name = os.path.basename(path)
            print(f"    Loading LoRA: {name} (weight={weight:.2f})")
            try:
                pipe.load_lora_weights(path)
                pipe.fuse_lora(lora_scale=weight)
                print(f"    Fused LoRA: {name}")
            except Exception as e:
                print(f"    WARNING: Failed to load/fuse LoRA {name}: {e}")

        self._tokenizer = pipe.tokenizer
        self._text_encoder = pipe.text_encoder
        self._scheduler = pipe.scheduler
        self._pipe = pipe

    def _reencode_prompts(self):
        """Re-encode all prompts with the current text encoder (after LoRA change)."""
        print(f"  Re-encoding {len(self._all_prompts)} prompt(s) with updated LoRA stack...")
        self._all_embeds = [self._encode_single(p) for p in self._all_prompts]
        with self._encode_lock:
            self._prompt_embeds = self._all_embeds[self._prompt_index]
            self._target_embeds = self._prompt_embeds.copy()

    def set_lora_stack(self, lora_stack):
        """Replace the LoRA stack at runtime and re-encode prompts."""
        self._lora_stack = list(lora_stack or [])
        print(f"  Updating LoRA stack: {self._lora_stack}")
        self._load_text_encoder_with_loras()
        self._reencode_prompts()
        # Release the heavy pipeline after fusing
        self._pipe = None
        import gc, torch
        gc.collect()
        torch.mps.empty_cache()
        print("  LoRA stack updated")
        return self._lora_stack

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
        """Full pipeline: preprocess → VAE enc → UNet → VAE dec → postprocess."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        r = self.process_frame_rgb(rgb)
        return cv2.cvtColor(r, cv2.COLOR_RGB2BGR)

    def process_frame_rgb(self, frame_rgb):
        """Same pipeline but accepts and returns RGB frames."""
        h, w = frame_rgb.shape[:2]
        if w > h:
            off = (w - h) // 2
            frame_rgb = frame_rgb[:, off:off + h]
        elif h > w:
            off = (h - w) // 2
            frame_rgb = frame_rgb[off:off + w, :]

        resized = cv2.resize(frame_rgb, (self.render_size, self.render_size))
        np.copyto(self._img_buf, self._norm_lut[resized].transpose(2, 0, 1)[np.newaxis])

        # Smooth prompt transition (thread-safe)
        with self._encode_lock:
            diff = self._target_embeds - self._prompt_embeds
            if np.abs(diff).max() > 1e-4:
                self._prompt_embeds = self._prompt_embeds + self._prompt_lerp_speed * diff
            prompt_embeds = self._prompt_embeds

        # VAE Encode
        enc = self.vae_encoder.predict({"image": self._img_buf})
        clean = np.array(enc["latent"]).astype(np.float16)

        # Latent feedback from previous frame
        if self._prev_denoised is not None and self.latent_feedback > 0:
            fb = np.float16(self.latent_feedback)
            clean = (1.0 - fb) * clean + fb * self._prev_denoised

        # Add fixed noise (same every frame for temporal coherence)
        noisy = self._sqrt_a * clean + self._sqrt_1ma * self._fixed_noise
        np.copyto(self._lat_buf, noisy)

        # UNet inference
        u = self.unet.predict({
            "sample": self._lat_buf, "timestep": self._t_buf,
            "encoder_hidden_states": prompt_embeds,
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
        return r
