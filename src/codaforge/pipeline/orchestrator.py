from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from codaforge.pipeline.agni_veda_pipeline import AgnivedaPipelineAdvisor
from codaforge.pipeline.checkpoint import PipelineCheckpoint
from codaforge.pipeline.extract_frames import FrameExtractor
from codaforge.pipeline.human_masks import HumanMaskGenerator
from codaforge.pipeline.multiview_keyframes import MultiviewKeyframeSelector
from codaforge.pipeline.quality_check import QualityChecker
from codaforge.pipeline.temporal_consistency import TemporalConsistencyFilter
from codaforge.pipeline.validation import PipelineValidator
from codaforge.reconstruction.ai_reconstruction import AIReconstructor
from codaforge.reconstruction.body_fitting import BodyFitter
from codaforge.reconstruction.body_volume_estimator import BodyVolumeEstimator
from codaforge.reconstruction.camera_calibration import CameraCalibrator
from codaforge.reconstruction.config import PipelineConfig, parse_quality_preset, detect_device
from codaforge.reconstruction.face_reconstruction import FaceReconstructor
from codaforge.reconstruction.gaussian_splatting import GaussianSplatting
from codaforge.reconstruction.monocular_depth import MonocularDepthEstimator
from codaforge.reconstruction.run_colmap import ColmapRunner
from codaforge.reconstruction.scale_estimation import ScaleEstimator
from codaforge.mesh.mesh_fusion import MeshFusion
from codaforge.mesh.mesh_cleanup import MeshCleanup
from codaforge.export.texture_baking import TextureBaker
from codaforge.export.export_manager import ExportManager
from codaforge.reports.report_generator import ReportGenerator
from codaforge.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class ForgePipeline:
    STAGES = [
        "quality_check",
        "extract_frames",
        "temporal_consistency",
        "depth_estimation",
        "scale_estimation",
        "multiview_keyframes",
        "human_masks",
        "body_volume_estimation",
        "run_colmap",
        "ai_reconstruction",
        "face_reconstruction",
        "gaussian_splatting",
        "mesh_fusion",
        "mesh_cleanup",
        "texture_baking",
        "export",
        "report",
    ]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = PipelineCheckpoint(
            str(self.output_dir), pipeline_id=config.pipeline_id
        )
        self.validator = PipelineValidator()

        self.advisor = AgnivedaPipelineAdvisor()

        preset = parse_quality_preset(config.quality)
        self.num_keyframes = config.num_keyframes or preset["num_keyframes"]

        self.results: dict[str, dict] = {}
        self.context: dict = {
            "video_path": config.video_path,
            "height_cm": config.height_cm,
            "weight_kg": config.weight_kg,
            "target_height_cm": config.target_height_cm,
            "quality": config.quality,
        }

    def run(self, resume: bool = False) -> dict:
        logger.info("Starting CODA Forge pipeline (v1.0.0)")
        logger.info("Config: height=%scm, weight=%skg, quality=%s",
                     self.config.height_cm, self.config.weight_kg, self.config.quality)
        logger.info("Device: %s", detect_device())

        if resume:
            last_stage = self.checkpoint.get_last_stage()
            if last_stage:
                resume_idx = self._stage_index(last_stage)
                if resume_idx >= 0:
                    logger.info("Resuming from stage %d: %s", resume_idx + 1, last_stage)
                    stages = self.STAGES[resume_idx + 1:]
                    for stage in self.STAGES[: resume_idx + 1]:
                        data = self.checkpoint.get_stage_data(stage)
                        if data:
                            self.results[stage] = data
                            self.context.update(data.get("context", {}))
                else:
                    stages = self.STAGES
            else:
                stages = self.STAGES
        else:
            stages = self.STAGES
            self.checkpoint.clear()

        for stage in stages:
            if not self._run_stage(stage):
                logger.error("Pipeline failed at stage: %s", stage)
                break

        report_gen = ReportGenerator(str(self.output_dir))
        report_result = report_gen.generate(self.results, self.config.__dict__)
        self.results["report"] = report_result
        self.checkpoint.save("report", {"report_path": report_result.get("json_path", "")})

        self._print_summary()
        return self.results

    def _run_stage(self, stage: str) -> bool:
        logger.info("=" * 60)
        logger.info("Stage %d/%d: %s", self._stage_index(stage) + 1, len(self.STAGES), stage)
        logger.info("=" * 60)

        try:
            stage_fn = getattr(self, f"_stage_{stage}", None)
            if stage_fn is None:
                logger.error("Unknown stage: %s", stage)
                return False

            result = stage_fn()
            self.results[stage] = result
            self.checkpoint.save(stage, {"result": result, "context": self.context})

            if not result.get("success", False):
                logger.warning("Stage %s completed with warnings", stage)
            return True

        except Exception as e:
            logger.exception("Stage %s failed: %s", stage, e)
            self.results[stage] = {"success": False, "error": str(e)}
            self.checkpoint.save(stage, {"error": str(e)})
            return False

    def _stage_index(self, stage: str) -> int:
        try:
            return self.STAGES.index(stage)
        except ValueError:
            return -1

    def _stage_quality_check(self) -> dict:
        checker = QualityChecker()
        quality_data = checker.analyze_video(self.config.video_path)
        self.context["quality_data"] = quality_data
        self.context["frames_dir"] = str(self.output_dir / "frames")

        advice = self.advisor.analyze_quality(quality_data)
        quality_score = quality_data.get("quality_score", 0.5)
        self.context["quality_score"] = quality_score

        return {
            "success": True,
            "quality_score": quality_score,
            "issues": quality_data.get("issues", []),
            "advice": advice,
            "duration_seconds": quality_data.get("duration_seconds", 0),
        }

    def _stage_extract_frames(self) -> dict:
        extractor = FrameExtractor(
            output_dir=str(self.output_dir / "frames"),
            target_fps=6,
        )
        frames = extractor.extract(self.config.video_path, max_frames=300)
        self.context["frames"] = frames
        self.context["num_frames"] = len(frames)
        return {"success": True, "num_frames": len(frames), "frames_dir": extractor.output_dir}

    def _stage_temporal_consistency(self) -> dict:
        frames = self.context.get("frames", [])
        if not frames:
            return {"success": False, "error": "No frames available"}

        filter_ = TemporalConsistencyFilter(
            output_dir=str(self.output_dir / "clean_frames")
        )
        clean = filter_.filter_frames(
            frames,
            depth_dir=str(self.output_dir / "depth") if self.context.get("depth_dir") else None,
        )
        self.context["clean_frames"] = clean
        self.context["num_clean_frames"] = len(clean)
        return {
            "success": True,
            "input_frames": len(frames),
            "output_frames": len(clean),
            "filtered": len(frames) - len(clean),
        }

    def _stage_depth_estimation(self) -> dict:
        frames = self.context.get("frames", [])
        if not frames:
            return {"success": False, "error": "No frames available"}

        strategy = self.advisor.suggest_depth_strategy(
            self.config.quality,
            has_gpu=self.config.use_gpu,
        )
        estimator = MonocularDepthEstimator(
            method=strategy,
            output_dir=str(self.output_dir / "depth"),
        )
        depth_maps = estimator.estimate_depth(frames)
        self.context["depth_dir"] = str(self.output_dir / "depth")
        self.context["depth_maps"] = depth_maps
        return {
            "success": True,
            "num_depth_maps": len(depth_maps),
            "method": strategy,
            "depth_dir": estimator.output_dir,
        }

    def _stage_scale_estimation(self) -> dict:
        frames = self.context.get("frames", [])
        estimator = ScaleEstimator(height_cm=self.config.height_cm)
        scale = estimator.estimate_scale(
            frames,
            masks=self.context.get("masks"),
        )
        self.context["scale_factor"] = scale
        return {"success": True, "scale_factor": scale, "height_cm": self.config.height_cm}

    def _stage_multiview_keyframes(self) -> dict:
        frames = self.context.get("clean_frames") or self.context.get("frames", [])
        if not frames:
            return {"success": False, "error": "No frames available"}

        selector = MultiviewKeyframeSelector(
            output_dir=str(self.output_dir / "keyframes"),
            num_keyframes=self.num_keyframes,
        )
        keyframes = selector.select_keyframes(frames)
        self.context["keyframes"] = keyframes
        return {
            "success": True,
            "num_keyframes": len(keyframes),
            "keyframes_dir": selector.output_dir,
        }

    def _stage_human_masks(self) -> dict:
        frames = self.context.get("frames", [])
        if not frames:
            return {"success": False, "error": "No frames available"}

        generator = HumanMaskGenerator(
            output_dir=str(self.output_dir / "masks"),
            method="rembg",
        )
        masks = generator.generate_masks(frames)
        self.context["masks"] = masks
        return {
            "success": True,
            "num_masks": len(masks),
            "masks_dir": generator.output_dir,
        }

    def _stage_body_volume_estimation(self) -> dict:
        masks = self.context.get("masks", [])
        depth_maps = self.context.get("depth_maps")

        estimator = BodyVolumeEstimator(
            height_cm=self.config.height_cm,
            weight_kg=self.config.weight_kg,
        )
        volume_data = estimator.estimate_volume(masks, depth_maps)
        self.context["volume_cm3"] = volume_data.get("volume_cm3", 0)
        return {"success": True, **volume_data}

    def _stage_run_colmap(self) -> dict:
        frames_dir = str(self.output_dir / "frames")
        runner = ColmapRunner(
            output_dir=str(self.output_dir / "colmap"),
            mode=self.config.colmap_mode,
        )
        result = runner.run(frames_dir)
        if result.get("success"):
            self.context["colmap_dir"] = result.get("colmap_dir", "")
            self.context["point_cloud"] = result.get("point_cloud", "")
        return result

    def _stage_ai_reconstruction(self) -> dict:
        colmap_dir = self.context.get("colmap_dir")
        frames = self.context.get("frames", [])

        reconstructor = AIReconstructor(
            output_dir=str(self.output_dir / "reconstruction")
        )
        result = reconstructor.reconstruct(
            colmap_dir=colmap_dir,
            frames=frames,
            depth_dir=self.context.get("depth_dir"),
        )
        if result.get("success"):
            self.context["reconstructed_mesh"] = result.get("mesh_path", "")
        return result

    def _stage_face_reconstruction(self) -> dict:
        frames = self.context.get("frames", [])
        keyframes = self.context.get("keyframes", [])

        reconstructor = FaceReconstructor(
            output_dir=str(self.output_dir / "face")
        )
        result = reconstructor.reconstruct_face(
            frames=frames,
            keyframes=keyframes,
            use_mediapipe=self.config.use_mediapipe,
        )
        if result.get("success"):
            self.context["face_mesh"] = result.get("mesh_path", "")
        return result

    def _stage_gaussian_splatting(self) -> dict:
        images_dir = str(self.output_dir / "frames")
        colmap_dir = self.context.get("colmap_dir")

        splat = GaussianSplatting(output_dir=str(self.output_dir / "splat"))
        result = splat.train(images_dir, colmap_dir)
        if result.get("success"):
            self.context["splat_file"] = result.get("splat_file", "")
        return result

    def _stage_mesh_fusion(self) -> dict:
        point_cloud = self.context.get("point_cloud")
        if not point_cloud or not os.path.isfile(point_cloud):
            for candidate in [
                str(self.output_dir / "splat" / "splat_fallback.ply"),
                str(self.output_dir / "colmap" / "fallback_points.ply"),
            ]:
                if os.path.isfile(candidate):
                    point_cloud = self.context["point_cloud"] = candidate
                    break

        if not point_cloud or not os.path.isfile(point_cloud):
            return {"success": False, "error": "No point cloud available"}

        fuser = MeshFusion(output_dir=str(self.output_dir / "mesh"))
        result = fuser.fuse(point_cloud, smooth=self.config.smooth_mesh)
        if result.get("success"):
            self.context["fusion_mesh"] = result.get("mesh_path", "")
        return result

    def _stage_mesh_cleanup(self) -> dict:
        mesh_path = (
            self.context.get("fusion_mesh")
            or self.context.get("reconstructed_mesh")
        )
        if not mesh_path or not os.path.isfile(mesh_path):
            return {"success": False, "error": "No mesh to clean"}

        cleaner = MeshCleanup(output_dir=str(self.output_dir / "mesh"))
        result = cleaner.clean(mesh_path)
        if result.get("success"):
            self.context["clean_mesh"] = result.get("mesh_path", "")
        return result

    def _stage_texture_baking(self) -> dict:
        mesh_path = self.context.get("clean_mesh") or self.context.get("fusion_mesh")
        if not mesh_path or not os.path.isfile(mesh_path):
            return {"success": False, "error": "No mesh to texture"}

        baker = TextureBaker(output_dir=str(self.output_dir / "textures"))
        result = baker.bake(
            mesh_path,
            frames=self.context.get("keyframes") or self.context.get("frames"),
        )
        if result.get("success"):
            self.context["textured_mesh"] = result.get("mesh_path", "")
        return result

    def _stage_export(self) -> dict:
        mesh_path = self.context.get("textured_mesh") or self.context.get("clean_mesh")
        if not mesh_path or not os.path.isfile(mesh_path):
            return {"success": False, "error": "No mesh to export"}

        tex_path = str(self.output_dir / "textures" / "texture.png")
        if not os.path.isfile(tex_path):
            tex_path = None

        exporter = ExportManager(
            output_dir=str(self.output_dir / "final")
        )
        result = exporter.export_all(
            mesh_path,
            texture_path=tex_path,
            formats=self.config.export_formats,
        )
        if result.get("success"):
            self.context["export_dir"] = result.get("output_dir", "")
        return result

    def _stage_report(self) -> dict:
        report_gen = ReportGenerator(str(self.output_dir / "final"))
        result = report_gen.generate(self.results, self.config.__dict__)
        return {"success": True, **result}

    def _print_summary(self) -> None:
        successful = sum(
            1 for r in self.results.values()
            if isinstance(r, dict) and r.get("success")
        )
        failed = sum(
            1 for r in self.results.values()
            if isinstance(r, dict) and not r.get("success")
        )
        logger.info("=" * 60)
        logger.info("Pipeline complete: %d/%d stages successful", successful, successful + failed)
        if failed:
            logger.warning("%d stage(s) failed", failed)
        logger.info("Output: %s", self.output_dir / "final")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CODA Forge 3D — Photorealistic human reconstruction from rotation video"
    )
    parser.add_argument("--video", required=True, help="Input rotation video path")
    parser.add_argument("--height", type=float, default=175.0, help="Subject height in cm")
    parser.add_argument("--weight", type=float, default=70.0, help="Subject weight in kg")
    parser.add_argument("--quality", choices=["fast", "balanced", "quality"],
                        default="balanced", help="Quality preset")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-similarity", action="store_true", help="Skip evaluation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    setup_logger("codaforge", level=level)

    config = PipelineConfig(
        video_path=args.video,
        height_cm=args.height,
        weight_kg=args.weight,
        quality=args.quality,
        output_dir=args.output_dir,
        run_similarity=not args.no_similarity,
        debug=args.debug,
    )

    pipeline = ForgePipeline(config)
    results = pipeline.run(resume=args.resume)

    final_report = results.get("report", {})
    if isinstance(final_report, dict):
        json_path = final_report.get("json_path", "")
        if json_path:
            print(f"\nReport saved to: {json_path}")

    success = any(
        r.get("success") for r in results.values()
        if isinstance(r, dict)
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
