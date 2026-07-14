from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TextureBaker:
    def __init__(self, output_dir: str = "textures"):
        self.output_dir = output_dir

    def bake(
        self,
        mesh_path: str,
        frames: list[str] | None = None,
    ) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)

        import trimesh
        mesh = trimesh.load(mesh_path)
        if mesh.is_empty or len(mesh.vertices) == 0 or mesh.vertices.shape[1] < 3 or not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            return {"success": False, "error": "Empty mesh or mesh has no faces"}

        uv_mesh, tex_array = self._generate_uv(mesh, frames)
        tex_path = os.path.join(self.output_dir, "texture.png")
        Image.fromarray(tex_array).save(tex_path)

        textured = self._create_textured_mesh(uv_mesh, tex_array)
        out_path = os.path.join(self.output_dir, "textured_mesh.glb")
        textured.export(out_path)

        return {
            "success": True,
            "mesh_path": out_path,
            "texture_path": tex_path,
            "vertices": len(textured.vertices),
        }

    def _generate_uv(
        self, mesh, frames: list[str] | None
    ) -> tuple:
        try:
            import xatlas
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            uv_mesh = mesh.__class__(
                vertices=mesh.vertices[vmapping],
                faces=indices,
                process=False,
            )
        except ImportError:
            uv_mesh = mesh.copy()
            _ = self._sphere_uv(mesh.vertices)

        tex_size = 1024
        texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)

        if frames:
            scores = []
            for fpath in frames:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                scores.append((fpath, laplacian, abs(brightness - 128)))

            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[: min(8, len(scores))]

            for fpath, _, _ in top:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (tex_size, tex_size))
                weight = 1.0 / len(top)
                texture = (texture * (1 - weight) + img_rgb * weight).astype(np.uint8)
        else:
            texture[:] = (200, 180, 160)

        return uv_mesh, texture

    def _sphere_uv(self, vertices: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vertices, axis=1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        dirs = vertices / norm
        u = 0.5 + np.arctan2(dirs[:, 2], dirs[:, 0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0)) / np.pi
        return np.column_stack([u, v])

    def _create_textured_mesh(self, uv_mesh, tex_array: np.ndarray):
        import trimesh

        if uv_mesh is None or uv_mesh.is_empty:
            return trimesh.Trimesh()

        tex_img = Image.fromarray(tex_array)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8,
        )

        uv = getattr(uv_mesh.visual, 'uv', None)
        if uv is None:
            uv = self._sphere_uv(uv_mesh.vertices)

        uv_mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_img, material=material)
        return uv_mesh
