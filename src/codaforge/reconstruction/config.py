from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass
class PipelineConfig:
    video_path: str = ""
    height_cm: float = 175.0
    weight_kg: float = 70.0
    quality: str = "balanced"
    output_dir: str = "output"
    fps: int = 30
    frames_dir: str = ""
    depth_method: str = "miDaS"
    depth_dir: str = ""
    colmap_path: str = "colmap"
    colmap_mode: str = "sparse"
    use_gpu: bool = False
    use_mediapipe: bool = True
    use_xatlas: bool = True
    num_keyframes: int = 12
    face_enhance: bool = True
    smooth_mesh: bool = True
    export_formats: tuple[str, ...] = ("glb", "obj", "stl", "ply")
    run_similarity: bool = False
    similarity_model: str = ""
    pipeline_id: str = ""
    debug: bool = False

    def __post_init__(self):
        if not self.video_path:
            raise ValueError("video_path is required")
        if not self.frames_dir:
            self.frames_dir = os.path.join(self.output_dir, "frames")
        if not self.depth_dir:
            self.depth_dir = os.path.join(self.output_dir, "depth")

    @property
    def target_height_cm(self) -> float:
        return self.height_cm

    @property
    def target_weight_kg(self) -> float:
        return self.weight_kg

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in keys}
        return cls(**filtered)


def get_config() -> PipelineConfig:
    return PipelineConfig()


def parse_quality_preset(preset: str) -> dict:
    presets = {
        "fast": {"colmap_mode": "sparse", "num_keyframes": 8, "depth_method": "miDaS"},
        "balanced": {"colmap_mode": "sparse", "num_keyframes": 12, "depth_method": "miDaS"},
        "quality": {"colmap_mode": "dense", "num_keyframes": 24, "depth_method": "depth_anything"},
    }
    return presets.get(preset, presets["balanced"])


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
