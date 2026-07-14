from __future__ import annotations

import logging
import os

import trimesh

logger = logging.getLogger(__name__)


class ExportManager:
    FORMAT_EXPORTERS = {
        "glb": "export_glb",
        "obj": "export_obj",
        "stl": "export_stl",
        "ply": "export_ply",
    }

    def __init__(self, output_dir: str = "export"):
        self.output_dir = output_dir

    def export_all(
        self,
        mesh_path: str,
        texture_path: str | None = None,
        formats: tuple[str, ...] = ("glb", "obj", "stl", "ply"),
    ) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        mesh = trimesh.load(mesh_path)
        if mesh.is_empty or len(mesh.vertices) == 0 or mesh.vertices.shape[1] < 3:
            return {"success": False, "error": "Empty mesh"}

        results = {}
        for fmt in formats:
            exporter = self.FORMAT_EXPORTERS.get(fmt)
            if exporter is None:
                logger.warning("Unsupported format: %s", fmt)
                continue
            out_path = os.path.join(self.output_dir, f"model.{fmt}")
            try:
                if fmt == "glb" and texture_path and os.path.isfile(texture_path):
                    self._export_textured_glb(mesh, texture_path, out_path)
                else:
                    getattr(self, exporter)(mesh, out_path)
                results[fmt] = {"success": True, "path": out_path}
            except Exception as e:
                results[fmt] = {"success": False, "error": str(e)}
                logger.error("Export %s failed: %s", fmt, e)

        return {
            "success": any(r.get("success") for r in results.values()),
            "exports": results,
            "output_dir": self.output_dir,
        }

    def export_glb(self, mesh: trimesh.Trimesh, path: str) -> None:
        mesh.export(path, file_type="glb")

    def _export_textured_glb(
        self, mesh: trimesh.Trimesh, texture_path: str, path: str
    ) -> None:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=None,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        )
        if mesh.visual.kind == "texture":
            mesh.visual.material = material
        mesh.export(path, file_type="glb")

    def export_obj(self, mesh: trimesh.Trimesh, path: str) -> None:
        mesh.export(path, file_type="obj")

    def export_stl(self, mesh: trimesh.Trimesh, path: str) -> None:
        mesh.export(path, file_type="stl")

    def export_ply(self, mesh: trimesh.Trimesh, path: str) -> None:
        mesh.export(path, file_type="ply")
