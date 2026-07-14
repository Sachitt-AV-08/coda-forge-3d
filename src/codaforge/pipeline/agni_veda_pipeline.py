"""Config-based pipeline advisor.

Replaces the original CODA Mind AGNI/VEDA agent system with a simple
rule-based advisor that recommends settings based on quality preset,
available hardware, and detected frame characteristics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AgnivedaPipelineAdvisor:
    def __init__(self, config_dir: str | None = None):
        self.config_dir = config_dir
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        return {
            "quality_presets": {
                "fast": {
                    "depth_strategy": "miDaS",
                    "msx_samples": 6,
                    "num_keyframes": 8,
                    "colmap_mode": "sparse",
                    "mesh_resolution": "low",
                },
                "balanced": {
                    "depth_strategy": "miDaS",
                    "msx_samples": 12,
                    "num_keyframes": 12,
                    "colmap_mode": "sparse",
                    "mesh_resolution": "medium",
                },
                "quality": {
                    "depth_strategy": "depth_anything",
                    "msx_samples": 24,
                    "num_keyframes": 24,
                    "colmap_mode": "dense",
                    "mesh_resolution": "high",
                },
            },
            "fallback_depth": "miDaS",
        }

    def get_quality_preset(self, quality_label: str) -> dict:
        return self.rules.get("quality_presets", {}).get(
            quality_label, self.rules["quality_presets"]["balanced"]
        )

    def suggest_depth_strategy(self, quality: str, has_gpu: bool = False) -> str:
        preset = self.get_quality_preset(quality)
        strategy = preset["depth_strategy"]
        if strategy == "depth_anything" and not has_gpu:
            return "miDaS"
        return strategy

    def suggest_mesh_repair(self, mesh_path: str) -> bool:
        if not os.path.isfile(mesh_path):
            return True
        try:
            import trimesh

            m = trimesh.load(mesh_path)
            if not m.is_watertight:
                return True
            if len(m.faces) < 100:
                return True
        except Exception:
            return True
        return False

    def suggest_face_strategy(self, has_gpu: bool, has_mediapipe: bool) -> str:
        if has_mediapipe:
            return "mediapipe_flame"
        return "parametric"

    def analyze_quality(self, quality_data: dict) -> dict:
        result = {"risk": "low", "recommendations": []}
        blur = quality_data.get("avg_blur", 1.0)
        brightness = quality_data.get("avg_brightness", 128)
        motion = quality_data.get("motion_score", 0.0)

        if blur < 0.3:
            result["risk"] = "high"
            result["recommendations"].append("Video is too blurry; consider a steady camera")
        if brightness < 40 or brightness > 220:
            result["risk"] = "medium"
            result["recommendations"].append("Poor lighting detected")
        if motion > 0.5:
            result["risk"] = "medium"
            result["recommendations"].append("Excessive motion; use a tripod")
        return result

    def log(self, msg: str, level: str = "info") -> None:
        getattr(logger, level, logger.info)(msg)
