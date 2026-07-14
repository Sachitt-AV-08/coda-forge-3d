from __future__ import annotations

import logging
import os

import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree

logger = logging.getLogger(__name__)


class MeshFusion:
    def __init__(self, output_dir: str = "mesh"):
        self.output_dir = output_dir

    def fuse(
        self,
        point_cloud_path: str,
        method: str = "poisson",
        smooth: bool = True,
    ) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        pcd = trimesh.load(point_cloud_path)
        if pcd.is_empty or len(pcd.vertices) < 10:
            return {"success": False, "error": "Point cloud too small"}

        points = pcd.vertices
        normals = self._estimate_normals(points)

        if method == "poisson" and len(points) > 100:
            mesh = self._poisson_recon(points, normals)
        elif method == "alpha":
            mesh = self._alpha_shape(points)
        else:
            mesh = self._delaunay_mesh(points)

        if mesh is None or mesh.is_empty:
            mesh = self._delaunay_mesh(points)

        if smooth and mesh is not None and not mesh.is_empty:
            mesh = self._laplacian_smooth(mesh)

        if mesh is not None and not mesh.is_empty:
            mesh_path = os.path.join(self.output_dir, "fusion_mesh.obj")
            mesh.export(mesh_path)
            return {
                "success": True,
                "mesh_path": mesh_path,
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "method": method,
                "is_watertight": mesh.is_watertight,
            }

        return {"success": False, "error": "Mesh generation failed"}

    def _estimate_normals(self, points: np.ndarray, k: int = 20) -> np.ndarray:
        tree = cKDTree(points)
        normals = np.zeros_like(points)
        for i, p in enumerate(points):
            _, idx = tree.query(p, k=k + 1)
            neighbors = points[idx[1:]]
            cov = np.cov(neighbors.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            if normal[1] < 0:
                normal = -normal
            normals[i] = normal
        return normals

    def _poisson_recon(self, points: np.ndarray, normals: np.ndarray) -> trimesh.Trimesh | None:
        try:
            return trimesh.smoothing.filter_poisson(
                trimesh.Trimesh(vertices=points, vertex_normals=normals)
            )
        except Exception:
            return None

    def _alpha_shape(self, points: np.ndarray, alpha: float = 0.05) -> trimesh.Trimesh | None:
        try:
            tri = Delaunay(points[:, :2])
            faces = []
            for simplex in tri.simplices:
                p1, p2, p3 = points[simplex]
                circum = self._circumradius(p1, p2, p3)
                if circum < alpha:
                    faces.append(simplex)
            if faces:
                return trimesh.Trimesh(vertices=points, faces=np.array(faces))
        except Exception:
            pass
        return None

    def _delaunay_mesh(self, points: np.ndarray) -> trimesh.Trimesh:
        tri = Delaunay(points[:, :2])
        return trimesh.Trimesh(vertices=points, faces=tri.simplices)

    def _laplacian_smooth(self, mesh: trimesh.Trimesh, iterations: int = 5) -> trimesh.Trimesh:
        return trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)

    @staticmethod
    def _circumradius(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        return a * b * c / (4 * area) if area > 0 else float("inf")
