#!/usr/bin/env python3
"""Extract the body of the most recent versioned section from an add-on CHANGELOG.md.

Usage:
    python3 scripts/extract_changelog_notes.py [changelog_path]

Defaults to ml_heating_underfloor/CHANGELOG.md when no path is supplied.
Prints the release body to stdout so it can be captured by CI for use as
a GitHub Release description.

Exit codes:
    0 — release body printed successfully
    1 — no versioned section found (prints a fallback message to stdout)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_VERSION_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+[^\]]*)\]")

DEFAULT_CHANGELOG = Path(__file__).parent.parent / "ml_heating_underfloor" / "CHANGELOG.md"


def extract_latest_release_notes(changelog_path: Path) -> str:
    """Return the body text of the most recent versioned section.

    Returns an empty string when the changelog cannot be parsed or the most
    recent versioned section has no content.
    """
    try:
        lines = changelog_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"::warning::Could not read {changelog_path}: {exc}", file=sys.stderr)
        return ""

    # Find the first versioned heading (e.g. ## [0.2.43] - 2026-05-16)
    start_index: int | None = None
    for i, line in enumerate(lines):
        if _VERSION_HEADING.match(line):
            start_index = i
            break

    if start_index is None:
        return ""

    # Collect lines until the next versioned heading or end-of-file
    body_lines: list[str] = []
    for line in lines[start_index + 1 :]:
        if _VERSION_HEADING.match(line):
            break
        body_lines.append(line)

    # Strip leading/trailing blank lines
    body = "\n".join(body_lines).strip()
    return body


def main() -> None:
    changelog_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHANGELOG

    body = extract_latest_release_notes(changelog_path)

    if body:
        print(body)
    else:
        # Emit a minimal placeholder so the release is still created cleanly
        print("No changelog entries found for this release.")
        sys.exit(1)


if __name__ == "__main__":
    main()
