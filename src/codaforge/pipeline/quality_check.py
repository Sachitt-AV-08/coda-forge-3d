import cv2
import numpy as np


class QualityChecker:
    def __init__(self):
        self.results = {}

    def analyze_video(self, video_path: str, sample_rate: int = 30) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        motion_scores = []
        blur_scores = []
        brightness_values = []
        prev_gray = None

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(blur)

                brightness = np.mean(gray)
                brightness_values.append(brightness)

                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    motion_scores.append(float(np.mean(mag)))
                prev_gray = gray
            frame_idx += 1

        cap.release()

        avg_blur = float(np.mean(blur_scores)) if blur_scores else 1.0
        avg_brightness = float(np.mean(brightness_values)) if brightness_values else 128.0
        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0
        is_rotation_video = avg_motion > 0.1

        self.results = {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "avg_blur": avg_blur,
            "avg_brightness": avg_brightness,
            "motion_score": avg_motion,
            "is_rotation_video": is_rotation_video,
            "quality_score": self._compute_quality_score(avg_blur, avg_brightness, avg_motion),
            "issues": self._detect_issues(avg_blur, avg_brightness, avg_motion, duration),
        }
        return self.results

    def _compute_quality_score(self, blur: float, brightness: float, motion: float) -> float:
        score = 1.0
        if blur < 0.3:
            score -= 0.3
        elif blur < 0.5:
            score -= 0.1
        if brightness < 40 or brightness > 220:
            score -= 0.2
        if motion > 0.5:
            score -= 0.2
        return max(0.0, score)

    def _detect_issues(self, blur: float, brightness: float, motion: float, duration: float) -> list[str]:
        issues = []
        if blur < 0.3:
            issues.append("blurry_video")
        if blur < 0.5:
            issues.append("slightly_blurry")
        if brightness < 40:
            issues.append("too_dark")
        if brightness > 220:
            issues.append("overexposed")
        if motion > 0.5:
            issues.append("excessive_motion")
        if duration < 10:
            issues.append("video_too_short")
        return issues

    def get_results(self) -> dict:
        return self.results
