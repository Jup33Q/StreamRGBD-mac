#!/usr/bin/env python3
"""RGBD pipeline: CoreML img2img + depth estimation."""
import os

import numpy as np
import cv2

from pipelines.coreml import Pipeline
from depth.estimators import DepthEstimator


def _crop_to_aspect(image_bgr, target_w, target_h):
    """Center-crop a BGR image to the target aspect ratio, then resize."""
    h, w = image_bgr.shape[:2]
    if w == target_w and h == target_h:
        return image_bgr
    cropped = _crop_to_aspect_no_resize(image_bgr, target_w, target_h)
    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return cropped


def _crop_to_aspect_no_resize(image_bgr, target_w, target_h):
    """Center-crop a BGR image to the target aspect ratio (no resize)."""
    h, w = image_bgr.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h
    if abs(current_ratio - target_ratio) < 1e-6:
        return image_bgr
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        off = (w - new_w) // 2
        return image_bgr[:, off:off + new_w]
    else:
        new_h = int(w / target_ratio)
        off = (h - new_h) // 2
        return image_bgr[off:off + new_h, :]


class RGBDPipeline(Pipeline):
    """CoreML img2img pipeline extended with RGBD output."""

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
        """
        Run img2img + depth estimation and return RGB, depth, and RGBD.

        Returns:
            result_bgr: output_height x output_width x 3 uint8 BGR image.
            depth_u8:   output_height x output_width uint8 depth map.
            rgbd:       output_height x output_width x 4 uint8 RGBD image.
        """
        # 1. Run the base CoreML img2img pipeline (square).
        result_bgr = self.process_frame(frame_bgr)

        if self.output_width != self.output_height:
            # 2. Crop to target aspect ratio at render resolution (e.g. 288x512).
            cropped_bgr = _crop_to_aspect_no_resize(result_bgr, self.output_width, self.output_height)
            rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)

            # 3. Estimate depth on the cropped AI RGB output.
            depth_u8 = self.depth_estimator.estimate(rgb)

            # 4. Upscale RGB and depth to final output resolution.
            result_bgr = cv2.resize(cropped_bgr, (self.output_width, self.output_height), interpolation=cv2.INTER_LANCZOS4)
            rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            depth_u8 = cv2.resize(depth_u8, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
        else:
            rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            depth_u8 = self.depth_estimator.estimate(rgb)

        # 5. Concatenate RGB and depth into a 4-channel RGBD frame.
        rgbd = np.concatenate([rgb, depth_u8[:, :, np.newaxis]], axis=2)

        return result_bgr, depth_u8, rgbd
