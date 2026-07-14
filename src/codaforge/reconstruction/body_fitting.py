from __future__ import annotations

import numpy as np
import trimesh


class BodyFitter:
    def __init__(self, height_cm: float = 175.0, weight_kg: float = 70.0):
        self.height_cm = height_cm
        self.weight_kg = weight_kg

    def fit_to_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if mesh.is_empty:
            return mesh

        current_height = self._measure_height(mesh)
        if current_height > 0:
            scale = self.height_cm / current_height
            mesh.vertices *= scale

        return mesh

    def _measure_height(self, mesh: trimesh.Trimesh) -> float:
        vertices = mesh.vertices
        if len(vertices) == 0:
            return 0.0
        return float(vertices[:, 1].max() - vertices[:, 1].min())

    def estimate_body_shape(self, masks: list[str]) -> dict:
        import cv2

        widths = []
        heights_list = []

        for mpath in masks:
            mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            rows = np.any(binary > 0, axis=1)
            cols = np.any(binary > 0, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
                x_min, x_max = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1
                widths.append(x_max - x_min)
                heights_list.append(y_max - y_min)

        return {
            "avg_width_px": float(np.mean(widths)) if widths else 0,
            "avg_height_px": float(np.mean(heights_list)) if heights_list else 0,
            "num_samples": len(widths),
        }
