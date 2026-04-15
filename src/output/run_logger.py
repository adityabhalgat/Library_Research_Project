"""Plain-text run logger for LLM requests and responses."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


class RunTextLogger:
    """Write structured plain-text logs for each pipeline run."""

    def __init__(
        self,
        run_type: str,
        logs_dir: str = "ogs",
        header_context: dict[str, Any] | None = None,
    ) -> None:
        self.run_type = run_type.strip().replace(" ", "_") or "run"
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now()
        ts = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.file_path = self.logs_dir / f"{ts}_{self.run_type}.log"

        self._write_line("=" * 100)
        self._write_line("LLM RUN LOG")
        self._write_line("=" * 100)
        self._write_line(f"Run Type      : {self.run_type}")
        self._write_line(f"Started At    : {self.started_at.isoformat()}")
        if header_context:
            for key, value in header_context.items():
                self._write_line(f"{key:<13}: {self._safe(value)}")
        self._write_line("")

    def _safe(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def _write_line(self, text: str = "") -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _write_block(self, title: str, body: str) -> None:
        self._write_line(f"--- {title} ---")
        self._write_line(body if body else "")
        self._write_line(f"--- END {title} ---")

    def log_project_header(self, project_id: str | None, project_title: str | None) -> None:
        self._write_line("#" * 100)
        self._write_line("PROJECT")
        self._write_line("#" * 100)
        if project_id:
            self._write_line(f"Project ID   : {project_id}")
        self._write_line(f"Project Title: {self._safe(project_title)}")
        self._write_line("")

    def log_llm_exchange(
        self,
        *,
        project_id: str | None,
        project_title: str | None,
        strategy_name: str,
        phase_name: str,
        prompt_text: str,
        response_text: str,
        has_image: bool,
    ) -> None:
        self._write_line("-" * 100)
        self._write_line(f"Timestamp    : {datetime.now().isoformat()}")
        if project_id:
            self._write_line(f"Project ID   : {project_id}")
        self._write_line(f"Project Title: {self._safe(project_title)}")
        self._write_line(f"Strategy     : {strategy_name}")
        self._write_line(f"Phase        : {phase_name}")
        self._write_line(f"Image Used   : {'Yes' if has_image else 'No'}")
        self._write_line("")
        self._write_block("PROMPT INPUT", self._safe(prompt_text))
        self._write_line("")
        self._write_block("LLM RESPONSE", self._safe(response_text))
        self._write_line("")

    def log_note(self, note: str) -> None:
        self._write_line(f"[NOTE] {note}")

    def close(self, summary: dict[str, Any] | None = None) -> None:
        self._write_line("")
        self._write_line("=" * 100)
        self._write_line("RUN SUMMARY")
        self._write_line("=" * 100)
        self._write_line(f"Ended At      : {datetime.now().isoformat()}")
        if summary:
            for key, value in summary.items():
                self._write_line(f"{key:<13}: {self._safe(value)}")
        self._write_line("")
