#!/usr/bin/env python3
"""RGBD pipeline: CoreML img2img + depth estimation."""
import os

import numpy as np
import cv2

from pipelines.coreml import Pipeline
from depth.estimators import DepthEstimator


class RGBDPipeline(Pipeline):
    """CoreML img2img pipeline extended with RGBD output."""

    def __init__(self, depth_model="auto", depth_backend="auto",
                 depth_coreml_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            result_bgr: output_size x output_size x 3 uint8 BGR image.
            depth_u8:   output_size x output_size uint8 depth map.
            rgbd:       output_size x output_size x 4 uint8 RGBD image.
        """
        # 1. Run the base CoreML img2img pipeline.
        result_bgr = self.process_frame(frame_bgr)

        # 2. Estimate depth on the AI RGB output.
        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        depth_u8 = self.depth_estimator.estimate(rgb)

        # 3. Concatenate RGB and depth into a 4-channel RGBD frame.
        rgbd = np.concatenate([rgb, depth_u8[:, :, np.newaxis]], axis=2)

        return result_bgr, depth_u8, rgbd
