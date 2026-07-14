from __future__ import annotations

import logging
import os

import numpy as np

from codaforge.reconstruction.config import detect_device

logger = logging.getLogger(__name__)


class GaussianSplatting:
    def __init__(self, output_dir: str = "splat"):
        self.output_dir = output_dir
        self.device = detect_device()

    def train(
        self,
        images_dir: str,
        colmap_dir: str | None = None,
        sparse_model: str = "0",
    ) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)
        result = self._try_official_splat(images_dir, colmap_dir, sparse_model)
        if result["success"]:
            return result
        return self._cpu_fallback(images_dir)

    def _try_official_splat(
        self, images_dir: str, colmap_dir: str | None, sparse_model: str
    ) -> dict:
        try:

            gs_dir = os.environ.get("GAUSSIAN_SPLATTING_DIR", "models/gaussian-splatting")
            train_script = os.path.join(gs_dir, "train.py")
            if not os.path.isfile(train_script):
                return {"success": False, "error": "train.py not found"}

            import subprocess
            import sys

            cmd = [
                sys.executable,
                train_script,
                "-s", images_dir,
                "-m", self.output_dir,
            ]
            if colmap_dir:
                cmd.extend(["--source_path", colmap_dir, "--model_path", sparse_model])

            subprocess.run(cmd, cwd=gs_dir, check=True, capture_output=True, timeout=3600)

            ply_path = os.path.join(self.output_dir, "point_cloud", "iteration_30000", "point_cloud.ply")
            if not os.path.isfile(ply_path):
                ply_path = os.path.join(self.output_dir, "point_cloud.ply")

            return {
                "success": True,
                "method": "official_3dgs",
                "output_dir": self.output_dir,
                "splat_file": ply_path,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _cpu_fallback(self, images_dir: str) -> dict:
        import cv2

        points = []
        colors = []
        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))]
        )[:20]

        for fname in image_files:
            fpath = os.path.join(images_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            sample = img_rgb[::10, ::10]
            for y in range(sample.shape[0]):
                for x in range(sample.shape[1]):
                    u = x * 10 + np.random.randint(0, 10)
                    v = y * 10 + np.random.randint(0, 10)
                    if u < w and v < h:
                        points.append([u, v, 0])
                        colors.append(sample[y, x] / 255.0)

        if not points:
            return {"success": False, "error": "No images processed"}

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)

        from scipy.spatial import Delaunay

        try:
            tri = Delaunay(points[:, :2])
            faces = tri.simplices
        except Exception:
            faces = np.array([])

        import trimesh
        mesh = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=colors)

        ply_path = os.path.join(self.output_dir, "splat_fallback.ply")
        mesh.export(ply_path)

        return {
            "success": True,
            "method": "cpu_fallback",
            "splat_file": ply_path,
            "num_points": len(points),
            "num_faces": len(faces),
        }
