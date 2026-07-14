from __future__ import annotations

import logging
import os

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class MeshCleanup:
    def __init__(self, output_dir: str = "mesh"):
        self.output_dir = output_dir

    def clean(self, mesh_path: str, watertight: bool = True) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        mesh = trimesh.load(mesh_path)
        if mesh.is_empty or len(mesh.vertices) == 0 or mesh.vertices.shape[1] < 3:
            return {"success": False, "error": "Empty mesh"}

        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            return {"success": False, "error": "Mesh has no faces"}

        mesh = self._remove_duplicates(mesh)
        mesh = self._remove_degenerate_faces(mesh)
        mesh = self._merge_vertices(mesh)
        mesh = self._flip_normals(mesh)

        if watertight and not mesh.is_watertight:
            mesh = self._make_watertight(mesh)

        clean_path = os.path.join(self.output_dir, "clean_mesh.obj")
        mesh.export(clean_path)

        return {
            "success": True,
            "mesh_path": clean_path,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
        }

    def _remove_duplicates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _remove_degenerate_faces(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        if len(mesh.faces) == 0:
            return mesh
        vertices = mesh.vertices
        faces = mesh.faces
        valid = []
        for face in faces:
            a, b, c = vertices[face]
            area = np.linalg.norm(np.cross(b - a, c - a)) / 2
            if area > 1e-10:
                valid.append(face)
        if valid:
            mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(valid))
        return mesh

    def _merge_vertices(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        return mesh

    def _flip_normals(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            mesh.fix_normals()
        except Exception:
            pass
        return mesh

    def _make_watertight(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            return trimesh.smoothing.filter_poisson(mesh)
        except Exception:
            try:
                return mesh.convex_hull
            except Exception:
                return mesh
