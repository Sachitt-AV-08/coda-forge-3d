from __future__ import annotations

import os
import cv2
import numpy as np
from pathlib import Path


class MultiviewKeyframeSelector:
    def __init__(self, output_dir: str = "keyframes", num_keyframes: int = 12):
        self.output_dir = output_dir
        self.num_keyframes = num_keyframes

    def select_keyframes(self, frames: list[str]) -> list[str]:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if len(frames) <= self.num_keyframes:
            return self._copy_all(frames)

        candidates = self._score_frames(frames)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[: self.num_keyframes]
        best.sort(key=lambda x: x[0])

        out_paths = []
        for idx, _ in best:
            src = frames[idx]
            dst = os.path.join(self.output_dir, os.path.basename(src))
            cv2.imwrite(dst, cv2.imread(src))
            out_paths.append(dst)
        return out_paths

    def _score_frames(self, frames: list[str]) -> list[tuple[int, float]]:
        scores = []
        prev_features = None

        for i, fpath in enumerate(frames):
            img = cv2.imread(fpath)
            if img is None:
                scores.append((i, 0.0))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            kp = self._detect_features(gray)
            feature_score = min(len(kp) / 500.0, 1.0)

            diversity = 0.0
            if prev_features is not None and len(kp) > 0 and len(prev_features) > 0:
                matches = self._match_features(prev_features, kp)
                if len(matches) > 0:
                    diversity = 1.0 - min(len(matches) / 100.0, 1.0)

            score = 0.4 * sharpness + 0.3 * feature_score + 0.3 * diversity
            scores.append((i, score))
            prev_features = kp

        return scores

    def _detect_features(self, gray: np.ndarray):
        orb = cv2.ORB_create(nfeatures=1000)
        return orb.detect(gray, None)

    def _match_features(self, kp1, kp2):
        orb = cv2.ORB_create(nfeatures=1000)
        desc1 = orb.compute(cv2.cvtColor(np.zeros((100, 100), dtype=np.uint8), cv2.COLOR_GRAY2BGR), kp1)
        desc2 = orb.compute(cv2.cvtColor(np.zeros((100, 100), dtype=np.uint8), cv2.COLOR_GRAY2BGR), kp2)
        if desc1[1] is None or desc2[1] is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1[1], desc2[1], k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def _copy_all(self, frames: list[str]) -> list[str]:
        out_paths = []
        for fpath in frames:
            dst = os.path.join(self.output_dir, os.path.basename(fpath))
            cv2.imwrite(dst, cv2.imread(fpath))
            out_paths.append(dst)
        return out_paths
