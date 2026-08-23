"""Tests for dashboard app utility helpers."""

import os
import sys
from unittest.mock import patch


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "dashboard")
)

from app_utils import safe_directory_file_count


def test_safe_directory_file_count_returns_count(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    assert safe_directory_file_count(str(tmp_path)) == 2


def test_safe_directory_file_count_returns_none_for_missing_dir(tmp_path):
    missing_dir = tmp_path / "missing"

    assert safe_directory_file_count(str(missing_dir)) is None


def test_safe_directory_file_count_returns_none_for_unreadable_dir(tmp_path):
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()

    with patch("app_utils.os.listdir", side_effect=PermissionError("denied")):
        assert safe_directory_file_count(str(existing_dir)) is None
