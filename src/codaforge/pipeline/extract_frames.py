import os
from pathlib import Path

import cv2
import numpy as np


class FrameExtractor:
    def __init__(self, output_dir: str = "frames", target_fps: int = 6):
        self.output_dir = output_dir
        self.target_fps = target_fps

    def extract(self, video_path: str, max_frames: int = 300) -> list[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30

        frame_interval = max(1, int(round(video_fps / self.target_fps)))

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved_idx = 0
        scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                score = self._sharpness_score(frame)
                scores.append((saved_idx, frame, score))
                saved_idx += 1
            frame_idx += 1

        cap.release()

        if not scores:
            return []

        scores.sort(key=lambda x: x[2], reverse=True)
        top_frames = scores[:min(max_frames, len(scores))]
        top_frames.sort(key=lambda x: x[0])

        out_paths = []
        for idx, frame, _ in top_frames:
            fname = f"frame_{idx:06d}.jpg"
            fpath = os.path.join(self.output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out_paths.append(fpath)

        self._write_manifest(out_paths)
        return out_paths

    def _sharpness_score(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _write_manifest(self, paths: list[str]) -> None:
        manifest_path = os.path.join(self.output_dir, "manifest.txt")
        with open(manifest_path, "w") as f:
            for p in paths:
                f.write(os.path.basename(p) + "\n")
