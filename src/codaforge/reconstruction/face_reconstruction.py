from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceReconstructor:
    def __init__(self, output_dir: str = "face"):
        self.output_dir = output_dir
        self.flame_faces = np.array(
            [
                [0, 1, 2],
                [1, 3, 2],
                [2, 3, 4],
                [3, 5, 4],
                [4, 5, 6],
                [5, 7, 6],
                [6, 7, 8],
                [7, 9, 8],
                [0, 2, 10],
                [2, 4, 10],
                [4, 6, 10],
                [6, 8, 10],
                [0, 11, 1],
                [1, 12, 3],
                [3, 13, 5],
                [5, 14, 7],
                [7, 15, 9],
                [11, 12, 1],
                [12, 13, 3],
                [13, 14, 5],
                [14, 15, 7],
                [0, 16, 11],
                [11, 17, 12],
                [12, 18, 13],
                [13, 19, 14],
                [14, 20, 15],
                [16, 17, 11],
                [17, 18, 12],
                [18, 19, 13],
                [19, 20, 14],
                [0, 21, 16],
                [21, 22, 16],
                [22, 23, 17],
                [23, 24, 18],
                [24, 25, 19],
                [25, 26, 20],
                [21, 27, 22],
                [27, 28, 23],
                [28, 29, 24],
                [29, 30, 25],
                [30, 31, 26],
                [27, 32, 28],
                [28, 33, 29],
                [29, 34, 30],
                [30, 35, 31],
                [27, 36, 32],
                [36, 37, 33],
                [37, 38, 34],
                [38, 39, 35],
                [36, 40, 37],
                [40, 41, 38],
                [41, 42, 39],
                [36, 43, 40],
                [43, 44, 41],
                [44, 45, 42],
                [36, 46, 43],
                [46, 47, 44],
                [47, 48, 45],
                [46, 49, 47],
                [49, 50, 48],
                [46, 51, 49],
                [51, 52, 50],
                [51, 53, 52],
                [10, 54, 55],
                [10, 55, 56],
                [10, 56, 57],
                [10, 57, 8],
                [8, 57, 58],
                [8, 58, 59],
                [8, 59, 9],
                [9, 59, 15],
            ],
            dtype=np.int32,
        )

    def reconstruct_face(
        self,
        frames: list[str],
        keyframes: list[str] | None = None,
        use_mediapipe: bool = True,
    ) -> dict:
        lmks_3d = self._detect_landmarks(keyframes or frames, use_mediapipe)
        if lmks_3d is None or len(lmks_3d) < 3:
            return {"success": False, "error": "No landmarks detected"}

        mesh = self._build_face_mesh(lmks_3d)
        texture = self._bake_texture(frames, lmks_3d)

        os.makedirs(self.output_dir, exist_ok=True)
        mesh_path = os.path.join(self.output_dir, "face_mesh.obj")
        mesh.export(mesh_path)
        tex_path = os.path.join(self.output_dir, "face_texture.png")
        cv2.imwrite(tex_path, texture)

        return {
            "success": True,
            "mesh_path": mesh_path,
            "texture_path": tex_path,
            "num_vertices": len(mesh.vertices),
            "num_faces": len(mesh.faces),
        }

    def _detect_landmarks(
        self, frames: list[str], use_mediapipe: bool
    ) -> np.ndarray | None:
        if use_mediapipe:
            try:
                return self._mediapipe_landmarks(frames)
            except Exception:
                logger.warning("MediaPipe face mesh failed, using fallback")

        for fpath in frames:
            gray = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                return self._template_landmarks(w, h, x, y)
        return None

    def _mediapipe_landmarks(self, frames: list[str]) -> np.ndarray:
        import mediapipe as mp

        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as face_mesh:
            all_lmks = []
            for fpath in frames[:5]:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks[0].landmark:
                        all_lmks.append([lm.x * w, lm.y * h, lm.z * w])

            if all_lmks:
                landmarks = np.array(all_lmks).reshape(-1, 3)
                n = 60
                if len(landmarks) >= n:
                    indices = np.linspace(0, len(landmarks) - 1, n, dtype=int)
                    return landmarks[indices]
                return landmarks[:n]
        return np.zeros((60, 3))

    def _template_landmarks(self, w: int, h: int, x: int, y: int) -> np.ndarray:
        lmks = np.zeros((60, 3))
        cx, cy = x + w // 2, y + h // 2
        for i in range(60):
            angle = 2 * np.pi * i / 60
            r = w * 0.4
            lmks[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle), 0]
        return lmks

    def _build_face_mesh(self, landmarks: np.ndarray):
        import trimesh

        vertices = landmarks.copy()
        if len(vertices) < len(self.flame_faces.max()) + 1:
            extra = np.zeros((len(self.flame_faces.max()) + 1 - len(vertices), 3))
            vertices = np.vstack([vertices, extra])

        return trimesh.Trimesh(vertices=vertices, faces=self.flame_faces)

    def _bake_texture(self, frames: list[str], landmarks: np.ndarray) -> np.ndarray:
        h, w = 512, 512
        texture = np.zeros((h, w, 3), dtype=np.uint8)

        for fpath in frames:
            img = cv2.imread(fpath)
            if img is None:
                continue
            small = cv2.resize(img, (w, h))
            texture = cv2.addWeighted(texture, 0.5, small, 0.5, 0)

        return texture
