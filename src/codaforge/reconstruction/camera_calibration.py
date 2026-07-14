from __future__ import annotations

import cv2
import numpy as np


class CameraCalibrator:
    def __init__(self):
        self.K: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.image_size: tuple[int, int] | None = None

    def calibrate_from_chessboard(
        self, image_paths: list[str], chessboard_size: tuple[int, int] = (9, 6), square_mm: float = 25.0
    ) -> dict:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_mm

        obj_points = []
        img_points = []

        for fpath in image_paths:
            img = cv2.imread(fpath)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                obj_points.append(objp)
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(refined)

        if not obj_points:
            return self._fallback_calibration(image_paths)

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        self.K = K
        self.dist_coeffs = dist
        self.image_size = gray.shape[::-1]

        return {
            "ret": ret,
            "K": K.tolist(),
            "dist": dist.tolist(),
            "rms": ret,
            "num_images": len(obj_points),
            "image_size": list(self.image_size),
        }

    def _fallback_calibration(self, image_paths: list[str]) -> dict:
        exif_data = self._read_exif(image_paths)
        if exif_data and "focal_length_mm" in exif_data:
            focal_px = exif_data["focal_length_mm"] * max(
                exif_data.get("width", 1920), exif_data.get("height", 1080)
            ) / 36.0
            cx = exif_data.get("width", 1920) / 2
            cy = exif_data.get("height", 1080) / 2
            self.K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=np.float64)
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            return {
                "K": self.K.tolist(),
                "dist": self.dist_coeffs.tolist(),
                "source": "exif",
                "focal_length_px": focal_px,
            }

        h, w = 1080, 1920
        if image_paths:
            sample = cv2.imread(image_paths[0])
            if sample is not None:
                h, w = sample.shape[:2]
        focal_px = max(w, h) * 1.2
        self.K = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return {
            "K": self.K.tolist(),
            "dist": self.dist_coeffs.tolist(),
            "source": "estimated",
            "focal_length_px": focal_px,
        }

    def _read_exif(self, image_paths: list[str]) -> dict:
        if not image_paths:
            return {}
        try:
            from PIL import Image

            img = Image.open(image_paths[0])
            exif = img._getexif()
            if exif:
                focal = exif.get(37386) or exif.get(41989)
                if focal:
                    w, h = img.size
                    return {"focal_length_mm": float(focal), "width": w, "height": h}
        except Exception:
            pass
        return {}

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.K is None:
            return image
        h, w = image.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist_coeffs, (w, h), 1, (w, h)
        )
        return cv2.undistort(image, self.K, self.dist_coeffs, None, new_K)

    def get_intrinsics(self) -> dict:
        if self.K is None:
            return {}
        return {"K": self.K.tolist(), "dist": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None}
