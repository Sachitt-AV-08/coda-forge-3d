import json
from pathlib import Path


class PipelineCheckpoint:
    def __init__(self, output_dir: str | Path, pipeline_id: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "pipeline_checkpoint.json"
        self.pipeline_id = pipeline_id

    def save(self, stage: str, data: dict | None = None) -> None:
        cp = self._load()
        cp["last_stage"] = stage
        if data:
            cp.setdefault("data", {})[stage] = data
        if self.pipeline_id:
            cp["pipeline_id"] = self.pipeline_id
        self.checkpoint_file.write_text(json.dumps(cp, indent=2, default=str), encoding="utf-8")

    def load(self) -> dict:
        return self._load()

    def get_last_stage(self) -> str | None:
        cp = self._load()
        return cp.get("last_stage")

    def get_stage_data(self, stage: str) -> dict | None:
        cp = self._load()
        return cp.get("data", {}).get(stage)

    def clear(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            try:
                return json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
