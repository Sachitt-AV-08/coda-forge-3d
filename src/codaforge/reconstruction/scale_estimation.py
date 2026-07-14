from __future__ import annotations

import numpy as np
import cv2


class ScaleEstimator:
    def __init__(self, height_cm: float = 175.0):
        self.height_cm = height_cm

    def estimate_scale(
        self, frames: list[str], masks: list[str] | None = None
    ) -> float:
        heights_px = []
        for fpath in frames:
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                _, _, w, _ = faces[0]
                head_width_cm = 16.0
                focal_px = w * self.height_cm / head_width_cm
                heights_px.append(focal_px)

        if masks:
            for mpath in masks:
                mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                rows = np.any(mask > 127, axis=1)
                if np.any(rows):
                    y_min = np.argmax(rows)
                    y_max = len(rows) - np.argmax(rows[::-1]) - 1
                    heights_px.append(y_max - y_min)

        if not heights_px:
            return 1.0

        avg_height_px = float(np.mean(heights_px))
        if avg_height_px <= 0:
            return 1.0

        scale_factor = self.height_cm / avg_height_px
        return scale_factor
