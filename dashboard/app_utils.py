"""Utility helpers for dashboard app rendering."""

import os
from typing import Optional


def safe_directory_file_count(directory: str) -> Optional[int]:
    """Return entry count for a directory, or None if it cannot be listed."""
    try:
        return len(os.listdir(directory))
    except OSError:
        return None
