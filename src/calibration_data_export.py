"""
calibration_data_export.py
--------------------------
Shared helper for exporting ML calibration training data as compressed CSV.

Both heating-correction and cooling calibration pipelines call
``export_training_data()`` after saving the model, so the training data
is available for offline Optuna HPO or Jupyter notebook analysis.

Export path: ``<unified_state_dir>/<model_kind>_training_data.csv.gz``
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

try:
    from . import config
except ImportError:
    try:
        import config  # type: ignore
    except ImportError:
        config = None  # type: ignore


def export_training_data(
    df_train: "pd.DataFrame",
    feature_cols: List[str],
    model_kind: str,
) -> Optional[str]:
    """Save training data (features + label) as compressed CSV next to the
    unified thermal state file for offline Optuna / notebook analysis.

    Parameters
    ----------
    df_train : DataFrame
        Complete training DataFrame (features + ``label`` column).
    feature_cols : list[str]
        Feature column names (label column is added automatically).
    model_kind : str
        ``"heating"`` or ``"cooling"`` — used as filename prefix.

    Returns
    -------
    str | None
        Path to the exported file, or ``None`` if export was skipped/failed.
    """
    try:
        state_file = getattr(config, "UNIFIED_STATE_FILE", "")
        if not state_file:
            logger.warning(
                "UNIFIED_STATE_FILE not set — skipping training data export"
            )
            return None

        export_dir = os.path.dirname(state_file) or "."
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(
            export_dir, f"{model_kind}_training_data.csv.gz"
        )

        # Deduplicate while preserving order; always include "label" last.
        seen: set = set()
        cols_to_save = []
        for c in (feature_cols + ["label"]):
            if c not in seen and c in df_train.columns:
                seen.add(c)
                cols_to_save.append(c)

        if not cols_to_save:
            logger.warning(
                "No exportable columns found in df_train — skipping export"
            )
            return None

        if "label" not in cols_to_save:
            logger.warning(
                "'label' column not in df_train — exported data will lack targets"
            )

        tmp_export = export_path + ".tmp"
        try:
            df_train[cols_to_save].to_csv(
                tmp_export, index=False, compression="gzip"
            )
            os.replace(tmp_export, export_path)
        except Exception:
            # Clean up any partial .tmp file before re-raising
            try:
                if os.path.exists(tmp_export):
                    os.remove(tmp_export)
            except OSError:
                pass
            raise

        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        logger.info(
            "=== Training data exported: %s (%.1f MB, %d rows, %d cols) ===",
            export_path,
            size_mb,
            len(df_train),
            len(cols_to_save),
        )
        return export_path
    except Exception as exc:
        logger.warning("Training data export failed: %s", exc)
        return None
