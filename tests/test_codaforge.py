from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from codaforge.pipeline.checkpoint import PipelineCheckpoint
from codaforge.pipeline.validation import PipelineValidator
from codaforge.pipeline.quality_check import QualityChecker
from codaforge.reconstruction.config import PipelineConfig, parse_quality_preset
from codaforge.reconstruction.model_registry import list_available
from codaforge.reconstruction.body_volume_estimator import BodyVolumeEstimator
from codaforge.reconstruction.scale_estimation import ScaleEstimator
from codaforge.reconstruction.camera_calibration import CameraCalibrator
from codaforge.reconstruction.parametric_body import generate_parametric_mesh
from codaforge.reconstruction.face_reconstruction import FaceReconstructor
from codaforge.reconstruction.body_fitting import BodyFitter
from codaforge.mesh.mesh_cleanup import MeshCleanup
from codaforge.mesh.mesh_fusion import MeshFusion
from codaforge.export.export_manager import ExportManager
from codaforge.export.texture_baking import TextureBaker
from codaforge.reports.report_generator import ReportGenerator
from codaforge.pipeline.agni_veda_pipeline import AgnivedaPipelineAdvisor
from codaforge.utils.system_check import run_system_check


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.height_cm == 175.0
        assert cfg.quality == "balanced"
        assert cfg.target_height_cm == 175.0

    def test_from_dict(self):
        cfg = PipelineConfig.from_dict({"height_cm": 180, "quality": "fast", "unknown": 42})
        assert cfg.height_cm == 180.0
        assert cfg.quality == "fast"

    def test_quality_presets(self):
        fast = parse_quality_preset("fast")
        assert fast["num_keyframes"] == 8

        quality = parse_quality_preset("quality")
        assert quality["num_keyframes"] == 24


class TestCheckpoint:
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            cp = PipelineCheckpoint(d)
            cp.save("stage1", {"key": "val"})
            assert cp.get_last_stage() == "stage1"
            data = cp.get_stage_data("stage1")
            assert data["key"] == "val"

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            cp = PipelineCheckpoint(d)
            cp.save("test")
            assert cp.checkpoint_file.exists()
            cp.clear()
            assert not cp.checkpoint_file.exists()


class TestValidator:
    def test_valid_input(self):
        v = PipelineValidator()
        ok = v.validate_stage_input("quality_check", {"video_path": "test.mp4"})
        assert ok

    def test_missing_input(self):
        v = PipelineValidator()
        ok = v.validate_stage_input("quality_check", {})
        assert not ok
        assert len(v.errors) > 0

    def test_positive_float(self):
        v = PipelineValidator()
        assert v.validate_positive_float(5.0, "test")
        assert not v.validate_positive_float(-1.0, "test")


class TestAdvisor:
    def test_quality_preset(self):
        a = AgnivedaPipelineAdvisor()
        fast = a.get_quality_preset("fast")
        assert fast["num_keyframes"] == 8

    def test_depth_strategy(self):
        a = AgnivedaPipelineAdvisor()
        assert a.suggest_depth_strategy("fast", False) == "miDaS"

    def test_analyze_quality(self):
        a = AgnivedaPipelineAdvisor()
        result = a.analyze_quality({"avg_blur": 0.2, "avg_brightness": 128, "motion_score": 0.0})
        assert result["risk"] == "high"


class TestModelRegistry:
    def test_list_available(self):
        available = list_available()
        assert isinstance(available, dict)
        for key in ("torch", "mediapipe", "rembg"):
            assert key in available


class TestBodyVolume:
    def test_no_masks(self):
        est = BodyVolumeEstimator(175.0, 70.0)
        result = est.estimate_volume([])
        assert result["volume_cm3"] == 0.0


class TestScaleEstimation:
    def test_no_frames(self):
        est = ScaleEstimator(175.0)
        assert est.estimate_scale([]) == 1.0


class TestParametricMesh:
    def test_generate(self):
        mesh = generate_parametric_mesh()
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_dimensions(self):
        mesh = generate_parametric_mesh(height_cm=175.0, weight_kg=70.0)
        assert not mesh.is_empty


class TestBodyFitter:
    def test_empty_mesh(self):
        import trimesh
        fitter = BodyFitter()
        empty = trimesh.Trimesh()
        result = fitter.fit_to_mesh(empty)
        assert result.is_empty


class TestFaceReconstruct:
    def test_no_frames(self):
        rec = FaceReconstructor(os.path.join(tempfile.mkdtemp(), "face"))
        result = rec.reconstruct_face([])
        assert not result["success"]


class TestCameraCalibrator:
    def test_no_images(self):
        cal = CameraCalibrator()
        result = cal.calibrate_from_chessboard([])
        assert "K" in result


class TestMeshCleanup:
    def test_empty_mesh(self):
        import trimesh
        mesh = trimesh.Trimesh()
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "test.obj")
        mesh.export(path)
        cleaner = MeshCleanup(tmp)
        result = cleaner.clean(path)
        assert not result["success"]


class TestMeshFusion:
    def test_insufficient_points(self):
        import trimesh
        tmp = tempfile.mkdtemp()
        pcd = trimesh.points.PointCloud(np.zeros((5, 3)))
        path = os.path.join(tmp, "test.ply")
        pcd.export(path)
        fuser = MeshFusion(tmp)
        result = fuser.fuse(path)
        assert not result["success"]


class TestExportManager:
    def test_empty_mesh(self):
        import trimesh
        tmp = tempfile.mkdtemp()
        mesh = trimesh.Trimesh()
        path = os.path.join(tmp, "in.obj")
        mesh.export(path)
        exporter = ExportManager(tmp)
        result = exporter.export_all(path, formats=("obj",))
        assert not result["success"]


class TestTextureBaker:
    def test_empty_mesh(self):
        import trimesh
        tmp = tempfile.mkdtemp()
        mesh = trimesh.Trimesh()
        path = os.path.join(tmp, "in.obj")
        mesh.export(path)
        baker = TextureBaker(tmp)
        result = baker.bake(path)
        assert not result["success"]


class TestReportGenerator:
    def test_generate_json(self):
        with tempfile.TemporaryDirectory() as d:
            gen = ReportGenerator(d)
            result = gen.generate({"test": {"success": True}})
            assert "json_path" in result
            assert os.path.isfile(result["json_path"])

    def test_html_output(self):
        with tempfile.TemporaryDirectory() as d:
            gen = ReportGenerator(d)
            result = gen.generate({"test": {"success": True}})
            assert os.path.isfile(result["html_path"])


class TestSystemCheck:
    def test_returns_dict(self):
        result = run_system_check(verbose=False)
        assert isinstance(result, dict)
