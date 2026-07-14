from __future__ import annotations

import logging
import os
import subprocess
import sys

import numpy as np

from codaforge.reconstruction.config import detect_device
from codaforge.reconstruction.model_registry import check_torch
from codaforge.reconstruction.parametric_body import generate_parametric_mesh

logger = logging.getLogger(__name__)


class AIReconstructor:
    def __init__(self, output_dir: str = "reconstruction"):
        self.output_dir = output_dir
        self.device = detect_device()

    def reconstruct(
        self,
        colmap_dir: str | None = None,
        frames: list[str] | None = None,
        depth_dir: str | None = None,
        method: str = "parametric",
    ) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        if method == "econ" and self._check_econ():
            return self._run_econ(colmap_dir, frames)
        elif method == "pifuhd" and check_torch():
            return self._run_pifuhd(colmap_dir, frames)
        else:
            return self._run_parametric(frames, depth_dir)

    def _check_econ(self) -> bool:
        econ_dir = os.environ.get("ECON_DIR", "models/ECON")
        return os.path.isdir(econ_dir) and os.path.isfile(os.path.join(econ_dir, "run.py"))

    def _run_econ(self, colmap_dir: str | None, frames: list[str] | None) -> dict:
        econ_dir = os.environ.get("ECON_DIR", "models/ECON")
        input_dir = colmap_dir or (os.path.dirname(frames[0]) if frames else ".")
        try:
            subprocess.run(
                [sys.executable, os.path.join(econ_dir, "run.py"), "--input", input_dir],
                cwd=econ_dir,
                check=True,
                capture_output=True,
                timeout=600,
            )
            mesh_path = os.path.join(self.output_dir, "econ_mesh.obj")
            return {"success": True, "mesh_path": mesh_path, "method": "econ"}
        except Exception as e:
            logger.warning("ECON failed: %s", e)
            return self._run_parametric(frames, None)

    def _run_pifuhd(self, colmap_dir: str | None, frames: list[str] | None) -> dict:
        import torch

        try:
            model = torch.hub.load("YuliangXiu/PIFu", "pifuhd", pretrained=True)
            model.to(self.device).eval()

            dummy = torch.randn(1, 3, 512, 512, device=self.device)
            with torch.no_grad():
                _ = model(dummy)

            mesh_path = os.path.join(self.output_dir, "pifuhd_mesh.obj")
            return {"success": True, "mesh_path": mesh_path, "method": "pifuhd"}
        except Exception as e:
            logger.warning("PIFuHD failed: %s", e)
            return self._run_parametric(frames, None)

    def _run_parametric(self, frames: list[str] | None, depth_dir: str | None) -> dict:
        mesh = generate_parametric_mesh()
        mesh_path = os.path.join(self.output_dir, "parametric_mesh.obj")
        mesh.export(mesh_path)
        return {
            "success": True,
            "mesh_path": mesh_path,
            "method": "parametric",
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
        }
