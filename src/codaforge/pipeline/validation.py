from pathlib import Path


class PipelineValidator:
    STAGE_REQUIREMENTS = {
        "quality_check": {"video_path"},
        "extract_frames": {"frames_dir"},
        "temporal_consistency": {"clean_frames_dir"},
        "depth_estimation": {"depth_dir"},
        "scale_estimation": {"scale_factor"},
        "multiview_keyframes": {"keyframes_dir"},
        "human_masks": {"masks_dir"},
        "body_volume_estimation": {"volume_cm3"},
        "run_colmap": {"colmap_dir", "point_cloud"},
        "ai_reconstruction": {"reconstructed_mesh"},
        "face_reconstruction": {"face_mesh"},
        "gaussian_splatting": {"splat_file"},
        "mesh_fusion": {"fusion_mesh"},
        "mesh_cleanup": {"clean_mesh"},
        "texture_baking": {"textured_mesh"},
        "export": {"export_dir"},
    }

    def __init__(self, logs: list[str] | None = None):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.logs = logs or []

    def validate_stage_input(self, stage: str, context: dict) -> bool:
        required = self.STAGE_REQUIREMENTS.get(stage, set())
        missing = [k for k in required if k not in context or context.get(k) is None]
        if missing:
            self.errors.append(f"Stage '{stage}': missing required inputs: {missing}")
            return False
        return True

    def validate_file_exists(self, path: str | Path, label: str = "file") -> bool:
        if not Path(path).exists():
            self.errors.append(f"{label} not found: {path}")
            return False
        return True

    def validate_dir_exists(self, path: str | Path, label: str = "directory") -> bool:
        if not Path(path).is_dir():
            self.errors.append(f"{label} not found: {path}")
            return False
        return True

    def validate_positive_float(self, value: float, name: str) -> bool:
        if value is None or value <= 0:
            self.errors.append(f"{name} must be positive, got {value}")
            return False
        return True

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> dict:
        return {"errors": self.errors, "warnings": self.warnings, "valid": len(self.errors) == 0}
