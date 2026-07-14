from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)

    def generate(
        self, pipeline_results: dict, config: dict | None = None
    ) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report = self._build_report(pipeline_results, config)
        json_path = self.output_dir / "coda_forge_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        html_path = self.output_dir / "coda_forge_report.html"
        html_path.write_text(self._generate_html(report), encoding="utf-8")

        return {
            "json_path": str(json_path),
            "html_path": str(html_path),
            "report": report,
        }

    def _build_report(self, results: dict, config: dict | None) -> dict:
        stages = {}
        for key, val in results.items():
            if isinstance(val, dict) and "success" in val:
                stages[key] = {
                    "success": val.get("success", False),
                    "outputs": {k: v for k, v in val.items() if k != "success"},
                }

        return {
            "pipeline_status": "completed" if stages else "failed",
            "stages": stages,
            "summary": {
                "total_stages": len(stages),
                "successful": sum(1 for s in stages.values() if s["success"]),
                "failed": sum(1 for s in stages.values() if not s["success"]),
            },
            "config": config or {},
        }

    def _generate_html(self, report: dict) -> str:
        stages_html = ""
        for name, data in report.get("stages", {}).items():
            status = "passed" if data["success"] else "failed"
            outputs = "<br>".join(
                f"<span class='key'>{k}:</span> {v}"[:100]
                for k, v in data.get("outputs", {}).items()
            )
            stages_html += f"""
            <div class="stage {status}">
                <div class="stage-name">{name}</div>
                <div class="stage-status">{status.upper()}</div>
                <div class="stage-outputs">{outputs}</div>
            </div>"""

        summary = report.get("summary", {})
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CODA Forge Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0f; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:2rem; }}
h1 {{ color:#00d4ff; margin-bottom:0.5rem; }}
.summary {{ display:flex; gap:2rem; margin:2rem 0; }}
.stat {{ background:#1a1a2e; padding:1rem 2rem; border-radius:8px; border:1px solid #2a2a4e; }}
.stat-value {{ font-size:2rem; font-weight:700; color:#00d4ff; }}
.stat-label {{ font-size:0.85rem; color:#888; }}
.stage {{ background:#1a1a2e; margin:0.5rem 0; padding:1rem; border-radius:6px; border-left:4px solid #555; display:grid; grid-template-columns:1fr 80px 2fr; gap:1rem; align-items:center; }}
.stage.passed {{ border-left-color:#00d4ff; }}
.stage.failed {{ border-left-color:#ff4444; }}
.stage-name {{ font-weight:600; }}
.stage-status {{ font-size:0.8rem; font-weight:700; text-align:center; }}
.stage.passed .stage-status {{ color:#00d4ff; }}
.stage.failed .stage-status {{ color:#ff4444; }}
.stage-outputs {{ font-size:0.8rem; color:#aaa; }}
.key {{ color:#00d4ff; }}
</style>
</head>
<body>
<h1>CODA Forge — Pipeline Report</h1>
<div class="summary">
<div class="stat"><div class="stat-value">{summary.get("successful", 0)}/{summary.get("total_stages", 0)}</div><div class="stat-label">Stages Passed</div></div>
<div class="stat"><div class="stat-value">{summary.get("failed", 0)}</div><div class="stat-label">Failed</div></div>
</div>
{stages_html}
</body>
</html>"""
