from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class BodyVolumeEstimator:
    def __init__(self, height_cm: float = 175.0, weight_kg: float = 70.0):
        self.height_cm = height_cm
        self.weight_kg = weight_kg

    def estimate_volume(
        self, masks: list[str], depth_maps: list[str] | None = None
    ) -> dict:
        slice_volumes = []
        for i, mask_path in enumerate(masks):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            area_px = np.sum(binary > 0)

            depth = 1.0
            if depth_maps and i < len(depth_maps):
                dm = cv2.imread(depth_maps[i], cv2.IMREAD_GRAYSCALE)
                if dm is not None:
                    depth_val = dm[binary > 0]
                    if len(depth_val) > 0:
                        depth = float(np.mean(depth_val)) / 255.0

            slice_volumes.append(area_px * depth)

        total_volume_px = sum(slice_volumes) if slice_volumes else 0
        height_px = self._estimate_height_px(masks)

        volume_cm3 = 0.0
        if height_px > 0 and total_volume_px > 0:
            scale = self.height_cm / height_px
            volume_cm3 = total_volume_px * (scale**3)

        return {
            "volume_cm3": volume_cm3,
            "height_px": height_px,
            "height_cm": self.height_cm,
            "num_slices": len(slice_volumes),
            "avg_slice_area_px": float(np.mean(slice_volumes)) if slice_volumes else 0,
        }

    def _estimate_height_px(self, masks: list[str]) -> float:
        heights = []
        for mpath in masks:
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            rows = np.any(mask > 127, axis=1)
            if np.any(rows):
                y_min = np.argmax(rows)
                y_max = len(rows) - np.argmax(rows[::-1]) - 1
                heights.append(y_max - y_min)
        return float(np.mean(heights)) if heights else 0.0
