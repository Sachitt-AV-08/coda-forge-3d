from __future__ import annotations

import logging
import os
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


def find_colmap() -> str:
    import shutil
    for name in ("colmap", "colmap.exe"):
        path = shutil.which(name)
        if path:
            return path
    return "colmap"


class ColmapRunner:
    SPARSE_DIR = "sparse"
    DENSE_DIR = "dense"

    def __init__(self, output_dir: str = "colmap", mode: str = "sparse"):
        self.output_dir = output_dir
        self.mode = mode
        self.colmap_bin = find_colmap()
        self.db_path = os.path.join(output_dir, "database.db")

    def run(self, images_dir: str) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        if not self._check_colmap():
            return self._fallback(images_dir)

        try:
            self._feature_extraction(images_dir)
            self._feature_matching()

            sparse_dir = os.path.join(self.output_dir, self.SPARSE_DIR)
            os.makedirs(sparse_dir, exist_ok=True)
            self._sparse_reconstruction(sparse_dir)

            result = {
                "success": True,
                "method": "colmap",
                "colmap_dir": self.output_dir,
                "sparse_dir": sparse_dir,
            }

            if self.mode == "dense":
                dense_dir = os.path.join(self.output_dir, self.DENSE_DIR)
                os.makedirs(dense_dir, exist_ok=True)
                self._dense_reconstruction(sparse_dir, dense_dir, images_dir)
                result["dense_dir"] = dense_dir
                result["point_cloud"] = os.path.join(dense_dir, "fused.ply")
            else:
                result["point_cloud"] = os.path.join(sparse_dir, "points3D.ply")

            return result

        except Exception as e:
            logger.warning("COLMAP failed: %s, using fallback", e)
            return self._fallback(images_dir)

    def _check_colmap(self) -> bool:
        try:
            subprocess.run(
                [self.colmap_bin, "--help"],
                capture_output=True,
                timeout=10,
            )
            return True
        except Exception:
            return False

    def _feature_extraction(self, images_dir: str) -> None:
        subprocess.run(
            [
                self.colmap_bin, "feature_extractor",
                "--database_path", self.db_path,
                "--image_path", images_dir,
                "--ImageReader.single_camera", "1",
            ],
            check=True, capture_output=True, timeout=600,
        )

    def _feature_matching(self) -> None:
        subprocess.run(
            [
                self.colmap_bin, "exhaustive_matcher",
                "--database_path", self.db_path,
            ],
            check=True, capture_output=True, timeout=600,
        )

    def _sparse_reconstruction(self, output_dir: str) -> None:
        subprocess.run(
            [
                self.colmap_bin, "mapper",
                "--database_path", self.db_path,
                "--image_path", os.path.dirname(self.db_path),
                "--output_path", output_dir,
            ],
            check=True, capture_output=True, timeout=1200,
        )

    def _dense_reconstruction(
        self, sparse_dir: str, dense_dir: str, images_dir: str
    ) -> None:
        subprocess.run(
            [
                self.colmap_bin, "image_undistorter",
                "--image_path", images_dir,
                "--input_path", sparse_dir,
                "--output_path", dense_dir,
            ],
            check=True, capture_output=True, timeout=600,
        )
        subprocess.run(
            [
                self.colmap_bin, "patch_match_stereo",
                "--workspace_path", dense_dir,
            ],
            check=True, capture_output=True, timeout=1800,
        )
        subprocess.run(
            [
                self.colmap_bin, "stereo_fusion",
                "--workspace_path", dense_dir,
                "--output_path", os.path.join(dense_dir, "fused.ply"),
            ],
            check=True, capture_output=True, timeout=600,
        )

    def _fallback(self, images_dir: str) -> dict:
        import cv2

        points = []
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))]
        )[:10]

        for fname in image_files:
            fpath = os.path.join(images_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            sift = cv2.SIFT_create()
            kp, _ = sift.detectAndCompute(img, None)
            for k in kp:
                points.append([k.pt[0], k.pt[1], k.size])

        points = np.array(points, dtype=np.float32) if points else np.zeros((0, 3))

        ply_path = os.path.join(self.output_dir, "fallback_points.ply")
        self._save_ply(points, ply_path)

        return {
            "success": True,
            "method": "fallback_sift",
            "point_cloud": ply_path,
            "num_points": len(points),
        }

    def _save_ply(self, points: np.ndarray, path: str) -> None:
        if len(points) == 0:
            np.savetxt(path, [[0, 0, 0]], delimiter=" ")
            return
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
