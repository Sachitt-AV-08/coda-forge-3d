"""Quickstart example for CODA Forge 3D.

Usage:
    python quickstart.py --video path/to/rotation_video.mp4
"""

from __future__ import annotations

import argparse

from codaforge.reconstruction.config import PipelineConfig
from codaforge.pipeline.orchestrator import ForgePipeline


def main():
    parser = argparse.ArgumentParser(description="CODA Forge 3D Quickstart")
    parser.add_argument("--video", required=True, help="Input rotation video")
    parser.add_argument("--height", type=float, default=175.0, help="Height in cm")
    parser.add_argument("--weight", type=float, default=70.0, help="Weight in kg")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        height_cm=args.height,
        weight_kg=args.weight,
        output_dir=args.output,
    )

    pipeline = ForgePipeline(config)
    results = pipeline.run()

    print(f"\nPipeline complete. Report: {results.get('report', {})}")


if __name__ == "__main__":
    main()
