# CODA Forge 3D

**Photorealistic 3D human reconstruction from phone rotation videos.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | quality_check | Lighting, blur, motion analysis |
| 2 | extract_frames | Adaptive sharpness-based frame selection |
| 3 | temporal_consistency | Optical flow + landmark smoothing + coverage |
| 4 | depth_estimation | Monocular depth (Depth Anything / MiDaS) |
| 5 | scale_estimation | Metric scale from body proportions |
| 6 | multiview_keyframes | Multi-view keyframe selection |
| 7 | human_masks | Background removal (RMBG / SAM) |
| 8 | body_volume_estimation | Horizontal slicing volume from masks + depth |
| 9 | run_colmap | COLMAP SfM → sparse/dense point cloud |
| 10 | ai_reconstruction | Implicit reconstruction (ECON / PIFuHD / SMPL-X) |
| 11 | face_reconstruction | Face mesh + UV texture (MediaPipe / FLAME) |
| 12 | gaussian_splatting | 3D Gaussian Splatting neural rendering |
| 13 | mesh_fusion | Poisson + alpha shape surface reconstruction |
| 14 | mesh_cleanup | Watertight mesh for 3D printing |
| 15 | texture_baking | xatlas UV unwrapping + multi-view blending |
| 16 | export | GLB / OBJ / PLY / STL export |
| 17 | report | HTML/PDF quality report |

## Quickstart

```bash
pip install coda-forge-3d

coda-forge --video input.mp4 --height 175
```

Or from source:

```bash
git clone https://github.com/coda-forge/coda-forge-3d.git
cd coda-forge-3d
pip install -e .

python -m codaforge.pipeline.orchestrator --video input.mp4 --height 175
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Input rotation video path |
| `--height` | 175 | Subject height in cm (for metric scale) |
| `--weight` | 70 | Subject weight in kg (for volume calibration) |
| `--quality` | balanced | Quality preset: `fast`, `balanced`, `quality` |
| `--output-dir` | output | Output directory |
| `--resume` | — | Resume from last checkpoint |
| `--no-similarity` | — | Skip evaluation stage |

## Output

```
output/
├── final/
│   ├── model.glb          # Textured 3D model
│   ├── model.obj           # Wavefront OBJ
│   ├── model.stl           # STL for 3D printing
│   ├── textured_mesh.glb   # High-res textured version
│   └── coda_forge_report.json
├── frames/                 # Extracted frames
├── depth/                  # Depth maps
├── masks/                  # Human masks
├── colmap/                 # COLMAP output
├── point_cloud.ply         # Dense point cloud
└── mesh/                   # Intermediate meshes
```

## Dependencies

### Required
- numpy, opencv-python, Pillow, trimesh, scipy, scikit-image, networkx, requests

### Optional (enhance quality)
- **COLMAP** — feature matching + SfM (`--quality quality`)
- **PyMCubes** — marching cubes surface extraction
- **xatlas** — UV unwrapping for texture baking
- **mediapipe** — face landmarks for face reconstruction
- **torch** + **torchvision** — depth estimation (Depth Anything / MiDaS)

## License

Apache 2.0
