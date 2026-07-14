from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path


class HumanMaskGenerator:
    METHODS = ("rembg", "backgroundremover", "sam")

    def __init__(self, output_dir: str = "masks", method: str = "rembg"):
        self.output_dir = output_dir
        self.method = method if method in self.METHODS else "rembg"

    def generate_masks(self, frames: list[str]) -> list[str]:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        mask_paths = []

        generator = self._get_generator()
        for fpath in frames:
            img = cv2.imread(fpath)
            if img is None:
                continue
            mask = generator(img)
            base = os.path.splitext(os.path.basename(fpath))[0]
            mask_path = os.path.join(self.output_dir, f"{base}_mask.png")
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)

        return mask_paths

    def _get_generator(self):
        if self.method == "rembg":
            return self._rembg_mask
        elif self.method == "sam":
            return self._sam_mask
        else:
            return self._rembg_mask  # fallback

    def _rembg_mask(self, image: np.ndarray) -> np.ndarray:
        try:
            from rembg import remove

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = remove(rgb)
            if out.shape[2] == 4:
                alpha = out[:, :, 3]
            else:
                alpha = np.ones(image.shape[:2], dtype=np.uint8) * 255
            _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            return mask
        except ImportError:
            return self._simple_bg_removal(image)

    def _sam_mask(self, image: np.ndarray) -> np.ndarray:
        try:
            from segment_anything import SamPredictor, sam_model_registry

            model = sam_model_registry["vit_b"]()
            predictor = SamPredictor(model)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)
            h, w = image.shape[:2]
            masks, _, _ = predictor.predict(
                point_coords=np.array([[w // 2, h // 2]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            mask = (masks[0] * 255).astype(np.uint8)
            return mask
        except Exception:
            return self._simple_bg_removal(image)

    def _simple_bg_removal(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask
