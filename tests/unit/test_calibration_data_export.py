"""
Unit tests for calibration_data_export.export_training_data().

Covers:
- Happy path: exports CSV.gz with correct columns, readable by pandas
- Missing UNIFIED_STATE_FILE: returns None, logs warning
- Exception handling: bad DataFrame → returns None, no crash
- Atomic write: no leftover .tmp file on success
- Column filtering: only existing columns are exported
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, feature_cols: list | None = None):
    """Create a simple training DataFrame with features + label."""
    if feature_cols is None:
        feature_cols = ["feat_a", "feat_b", "feat_c"]
    rng = np.random.RandomState(42)
    data = {col: rng.randn(n) for col in feature_cols}
    data["label"] = rng.randn(n)
    return pd.DataFrame(data), feature_cols


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportTrainingData:
    """Tests for the export_training_data() function."""

    def test_happy_path_creates_readable_csv_gz(self, tmp_path):
        """Exported CSV.gz can be read back with correct shape and columns."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(200)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is not None
        assert os.path.isfile(result)
        assert result.endswith("heating_training_data.csv.gz")

        # Read back and verify
        df_back = pd.read_csv(result)
        assert list(df_back.columns) == feat_cols + ["label"]
        assert len(df_back) == 200

    def test_cooling_filename(self, tmp_path):
        """model_kind='cooling' produces the right filename."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(50)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, feat_cols, "cooling")

        assert result is not None
        assert "cooling_training_data.csv.gz" in result

    def test_missing_unified_state_file_returns_none(self):
        """Returns None and logs warning when UNIFIED_STATE_FILE is empty."""
        cfg = SimpleNamespace(UNIFIED_STATE_FILE="")

        df, feat_cols = _make_df(10)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is None

    def test_config_none_returns_none(self):
        """Returns None when config is None (no UNIFIED_STATE_FILE attr)."""
        df, feat_cols = _make_df(10)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", None):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is None

    def test_no_leftover_tmp_file(self, tmp_path):
        """After successful export, no .tmp file remains."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(50)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            mod.export_training_data(df, feat_cols, "heating")

        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_column_filtering_skips_missing(self, tmp_path):
        """Columns not in df_train are silently excluded."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, _ = _make_df(50, feature_cols=["feat_a", "feat_b"])
        # Request a column that doesn't exist
        requested_cols = ["feat_a", "feat_b", "nonexistent_col"]

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, requested_cols, "heating")

        assert result is not None
        df_back = pd.read_csv(result)
        assert "nonexistent_col" not in df_back.columns
        assert list(df_back.columns) == ["feat_a", "feat_b", "label"]

    def test_exception_returns_none(self, tmp_path):
        """Export failure (e.g. read-only dir) returns None, doesn't crash."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(10)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg), \
             patch.object(mod.os, "makedirs", side_effect=PermissionError("denied")):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is None

    def test_overwrites_previous_export(self, tmp_path):
        """A second export overwrites the first file cleanly."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        import src.calibration_data_export as mod

        df1, feat_cols = _make_df(100)
        with patch.object(mod, "config", cfg):
            mod.export_training_data(df1, feat_cols, "heating")

        df2, _ = _make_df(50)
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df2, feat_cols, "heating")

        df_back = pd.read_csv(result)
        assert len(df_back) == 50  # second export, not 100

    def test_tmp_file_cleaned_up_on_os_replace_failure(self, tmp_path):
        """If os.replace raises, the .tmp file is removed before returning None."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(20)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg), \
             patch.object(mod.os, "replace", side_effect=OSError("replace failed")):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is None
        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == []

    def test_label_in_feature_cols_not_duplicated(self, tmp_path):
        """If 'label' is already in feature_cols, it appears only once in export."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        # Include 'label' explicitly in feature_cols (defensive edge case)
        feat_cols_with_label = ["feat_a", "feat_b", "label"]
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "feat_a": rng.randn(30),
            "feat_b": rng.randn(30),
            "label": rng.randn(30),
        })

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, feat_cols_with_label, "heating")

        assert result is not None
        df_back = pd.read_csv(result)
        assert list(df_back.columns).count("label") == 1

    def test_empty_feature_cols_returns_none(self, tmp_path):
        """Returns None and logs warning when no columns match df_train."""
        state_file = str(tmp_path / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df = pd.DataFrame({"label": [1.0, 2.0]})  # no feature columns

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            # feature_cols references columns that don't exist; label is also absent
            # from the feature list so cols_to_save = ["label"] — NOT empty.
            # Use a df with no columns at all to trigger the empty guard.
            empty_df = pd.DataFrame()
            result = mod.export_training_data(empty_df, [], "heating")

        assert result is None

    def test_creates_export_directory(self, tmp_path):
        """Export dir is created if it doesn't exist."""
        export_dir = tmp_path / "subdir" / "nested"
        state_file = str(export_dir / "unified_thermal_state.json")
        cfg = SimpleNamespace(UNIFIED_STATE_FILE=state_file)

        df, feat_cols = _make_df(30)

        import src.calibration_data_export as mod
        with patch.object(mod, "config", cfg):
            result = mod.export_training_data(df, feat_cols, "heating")

        assert result is not None
        assert os.path.isdir(str(export_dir))
