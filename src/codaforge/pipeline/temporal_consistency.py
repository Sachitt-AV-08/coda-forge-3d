from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path


class TemporalConsistencyFilter:
    def __init__(self, output_dir: str = "clean_frames"):
        self.output_dir = output_dir
        self.flow_cache = {}

    def filter_frames(
        self, frames: list[str], depth_dir: str | None = None
    ) -> list[str]:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if len(frames) < 2:
            for src in frames:
                dst = os.path.join(self.output_dir, os.path.basename(src))
                cv2.imwrite(dst, cv2.imread(src))
            return frames

        clean_paths = []
        prev_gray = None
        valid_indices = []

        for i, fpath in enumerate(frames):
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                if np.mean(mag) > 10.0:
                    prev_gray = gray
                    continue

            prev_gray = gray
            valid_indices.append(i)

        if depth_dir:
            valid_indices = self._apply_coverage_filter(frames, valid_indices, depth_dir)

        for idx in valid_indices:
            src = frames[idx]
            dst = os.path.join(self.output_dir, os.path.basename(src))
            cv2.imwrite(dst, cv2.imread(src))
            clean_paths.append(dst)

        return clean_paths

    def _apply_coverage_filter(
        self, frames: list[str], indices: list[int], depth_dir: str
    ) -> list[int]:
        valid = []
        coverage_scores = []
        for idx in indices:
            base = os.path.splitext(os.path.basename(frames[idx]))[0]
            depth_path = os.path.join(depth_dir, f"{base}_depth.png")
            if not os.path.isfile(depth_path):
                valid.append(idx)
                continue
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                valid.append(idx)
                continue
            score = float(np.std(depth))
            coverage_scores.append((idx, score))

        if not coverage_scores:
            return indices

        coverage_scores.sort(key=lambda x: x[1], reverse=True)
        n = max(len(coverage_scores) // 2, min(8, len(coverage_scores)))
        selected = coverage_scores[:n]
        selected.sort(key=lambda x: x[0])
        return [idx for idx, _ in selected]

    def smooth_landmarks(
        self, landmarks: list[np.ndarray], window: int = 5
    ) -> list[np.ndarray]:
        if not landmarks:
            return landmarks
        smoothed = []
        n = len(landmarks)
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            avg = np.mean(landmarks[start:end], axis=0)
            smoothed.append(avg)
        return smoothed
