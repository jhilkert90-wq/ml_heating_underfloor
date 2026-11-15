"""Live export of the offline trajectory-control comparison, one cycle at a time.

A logging handler collects the formatted log lines of each control cycle. When
the next ``CYCLE n START`` line appears, the finished cycle is handed to
``python -m validation.compare_trajectory_control --append-block`` in a separate,
low-priority subprocess (the comparison patches ``config`` and must never touch
the live process). The first run creates the CSV/JSON/log files next to the
unified thermal state; later runs only update them.

The cycle details are DEBUG log lines, so the add-on must run with debug
logging; otherwise cycles are skipped (and reported once).

Environment:
    TRAJECTORY_COMPARISON_EXPORT      true/false (default true)
    TRAJECTORY_COMPARISON_MAX_ROWS    newest CSV rows kept (default 5000)
    TRAJECTORY_COMPARISON_COMFORT_BAND  K (default 0.1)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

_START = re.compile(r"CYCLE \d+ START:")
_MAX_BLOCK_CHARS = 8_000_000
_TIMEOUT_SECONDS = 300
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_ROOT = Path(__file__).resolve().parents[1]


def _enabled() -> bool:
    return os.getenv("TRAJECTORY_COMPARISON_EXPORT", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


class LiveComparisonExporter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(logging.Formatter(_LOG_FORMAT))
        self._lines: list[str] = []
        self._chars = 0
        self._in_cycle = False
        self._overflow = False
        self._busy = threading.Event()
        self._warned_no_debug = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            if _START.search(record.getMessage()):
                self._dispatch()
                self._lines, self._chars = [], 0
                self._in_cycle, self._overflow = True, False
            if not self._in_cycle or self._overflow:
                return
            self._lines.append(line)
            self._chars += len(line) + 1
            if self._chars > _MAX_BLOCK_CHARS:
                self._overflow = True
                self._lines = []
        except Exception:  # a log handler must never break the control loop
            pass

    def _dispatch(self) -> None:
        if not self._in_cycle or self._overflow or not self._lines:
            return
        if len(self._lines) < 5 and not self._warned_no_debug:
            self._warned_no_debug = True
            logging.getLogger(__name__).warning(
                "Live comparison export: cycle log has almost no DEBUG lines; "
                "enable debug logging for the export to work."
            )
        if self._busy.is_set():
            logging.getLogger(__name__).debug(
                "Live comparison export still running; skipping one cycle"
            )
            return
        block = "\n".join(self._lines) + "\n"
        self._busy.set()
        threading.Thread(
            target=self._run, args=(block,), daemon=True,
            name="live-comparison-export",
        ).start()

    def _run(self, block: str) -> None:
        log = logging.getLogger(__name__)
        try:
            command = [
                sys.executable, "-m", "validation.compare_trajectory_control",
                "--append-block",
                "--comfort-band", os.getenv("TRAJECTORY_COMPARISON_COMFORT_BAND", "0.1"),
                "--max-rows", os.getenv("TRAJECTORY_COMPARISON_MAX_ROWS", "5000"),
            ]
            result = subprocess.run(
                command, input=block, text=True, capture_output=True,
                cwd=_ROOT, timeout=_TIMEOUT_SECONDS,
                preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None,
            )
            if result.returncode != 0:
                log.warning(
                    "Live comparison export failed (rc=%s): %s",
                    result.returncode, (result.stderr or "")[-500:],
                )
        except Exception as exc:
            log.warning("Live comparison export error: %s", exc)
        finally:
            self._busy.clear()


def install_live_export() -> LiveComparisonExporter | None:
    """Attach the exporter to the root logger (idempotent)."""
    if not _enabled():
        return None
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, LiveComparisonExporter):
            return handler
    if not (_ROOT / "validation" / "compare_trajectory_control.py").exists():
        logging.getLogger(__name__).warning(
            "Live comparison export disabled: validation/ not found in %s", _ROOT
        )
        return None
    exporter = LiveComparisonExporter()
    root.addHandler(exporter)
    if root.getEffectiveLevel() > logging.DEBUG:
        logging.getLogger(__name__).warning(
            "Live comparison export needs DEBUG logging; cycles will be skipped."
        )
    return exporter
