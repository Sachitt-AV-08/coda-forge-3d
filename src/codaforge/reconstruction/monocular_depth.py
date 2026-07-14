from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


class MonocularDepthEstimator:
    def __init__(self, method: str = "miDaS", output_dir: str = "depth"):
        self.method = method
        self.output_dir = output_dir

    def estimate_depth(self, frames: list[str]) -> list[str]:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        estimator = self._get_estimator()
        depth_paths = []

        for fpath in frames:
            img = cv2.imread(fpath)
            if img is None:
                continue
            depth_map = estimator(img)
            base = os.path.splitext(os.path.basename(fpath))[0]
            depth_path = os.path.join(self.output_dir, f"{base}_depth.png")
            cv2.imwrite(depth_path, depth_map)
            depth_paths.append(depth_path)

        return depth_paths

    def _get_estimator(self):
        if self.method == "depth_anything":
            return self._depth_anything_estimator
        return self._midas_estimator

    def _midas_estimator(self, image: np.ndarray) -> np.ndarray:
        try:
            import torch

            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            model.eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb)
            with torch.no_grad():
                prediction = model(input_tensor)
            depth = prediction.squeeze().cpu().numpy()
        except Exception:
            depth = self._simple_depth(image)

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return (depth * 255).astype(np.uint8)

    def _depth_anything_estimator(self, image: np.ndarray) -> np.ndarray:
        try:
            import torch

            model = torch.hub.load("DepthAnything/Depth-Anything-V2", "depth_anything_v2_vitl")
            model.eval()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                depth = model(input_tensor).squeeze().cpu().numpy()
        except Exception:
            depth = self._simple_depth(image)

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return (depth * 255).astype(np.uint8)

    def _simple_depth(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        depth = cv2.Laplacian(blur, cv2.CV_64F)
        depth = np.abs(depth)
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        return depth
