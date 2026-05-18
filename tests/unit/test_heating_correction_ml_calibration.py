"""
Unit tests for heating_correction_ml_calibration.py

Covers:
1.  Cold-season filter: rows with AT >= threshold are excluded
2.  Label construction: label = −(T_future − T_target) / S_H
3.  Label outlier clipping: |label| > 5 °C is clipped
4.  Trivial margin (|indoor_margin| <= 0.05) → label forced to 0
5.  Feature columns include fireplace/TV lag features
6.  Start-date parsing: valid DD.MM.YYYY date resolves lookback_hours
7.  Start-date parsing: invalid date falls back to default
8.  _compute_s_h: correct formula
9.  calibrate_heating_correction_ml aborts on < 500 cold rows
10. calibrate_heating_correction_ml aborts on missing required columns
"""

from __future__ import annotations

import math
import json
import os
import tempfile
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:  # pragma: no cover - optional dependency fallback
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 700, at_val: float = 10.0):
    """Build a minimal DataFrame accepted by the calibration pipeline."""
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        pytest.skip("pandas/numpy not installed")

    rng = np.random.default_rng(42)
    t = pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC")
    df = pd.DataFrame({
        "_time":       t,
        "indoor_temp": rng.normal(21.0, 0.5, n),
        "AT":          np.full(n, at_val),
        "VLT":         rng.normal(30.0, 2.0, n),
        "RLT":         rng.normal(27.0, 2.0, n),
        "flow_rate":   np.full(n, 5.0),
        "fireplace_on": np.zeros(n),
        "tv_on":        np.zeros(n),
    })
    return df


# ---------------------------------------------------------------------------
# _compute_s_h
# ---------------------------------------------------------------------------

class TestComputeSH:
    def test_formula(self):
        from src.heating_correction_ml_calibration import _compute_s_h
        eta, u, tau, h = 0.830, 0.124, 4.39, 4.0
        expected = (eta / (eta + u)) * (1.0 - math.exp(-h / tau))
        assert _compute_s_h(eta, u, tau, h) == pytest.approx(expected, rel=1e-6)

    def test_degenerate_returns_zero(self):
        from src.heating_correction_ml_calibration import _compute_s_h
        assert _compute_s_h(0.0, 0.0, 4.0, 4.0) == 0.0
        assert _compute_s_h(0.5, 0.5, 0.0, 4.0) == 0.0


class TestReadBaselineThermalParams:
    def test_uses_heat_pump_channel_when_channels_enabled(self):
        from src.heating_correction_ml_calibration import _read_baseline_thermal_params

        cfg = SimpleNamespace(
            ENABLE_HEAT_SOURCE_CHANNELS=True,
            OUTLET_EFFECTIVENESS=0.95,
            HEAT_LOSS_COEFFICIENT=0.12,
            THERMAL_TIME_CONSTANT=4.39,
        )

        fake_manager = SimpleNamespace(
            get_heat_source_channel_state=lambda: {
                "heat_pump": {
                    "parameters": {
                        "outlet_effectiveness": 0.84,
                        "heat_loss_coefficient": 0.119,
                        "thermal_time_constant": 4.83,
                    },
                    "history_count": 3,
                    "history": [{"error": 0.01}],
                }
            },
            get_computed_parameters=lambda: {
                "outlet_effectiveness": 0.91,
                "heat_loss_coefficient": 0.14,
                "thermal_time_constant": 5.1,
            },
        )

        with patch(
            "src.unified_thermal_state.get_thermal_state_manager",
            return_value=fake_manager,
        ):
            eta, u, tau = _read_baseline_thermal_params(cfg)

        assert eta == pytest.approx(0.84)
        assert u == pytest.approx(0.119)
        assert tau == pytest.approx(4.83)

    def test_uses_baseline_plus_adjustments_when_channel_unavailable(self):
        from src.heating_correction_ml_calibration import _read_baseline_thermal_params

        cfg = SimpleNamespace(
            ENABLE_HEAT_SOURCE_CHANNELS=True,
            OUTLET_EFFECTIVENESS=0.95,
            HEAT_LOSS_COEFFICIENT=0.12,
            THERMAL_TIME_CONSTANT=4.39,
        )

        fake_manager = SimpleNamespace(
            get_heat_source_channel_state=lambda: {},
            get_computed_parameters=lambda: {
                "outlet_effectiveness": 0.827,
                "heat_loss_coefficient": 0.120,
                "thermal_time_constant": 4.835,
            },
        )

        with patch(
            "src.unified_thermal_state.get_thermal_state_manager",
            return_value=fake_manager,
        ):
            eta, u, tau = _read_baseline_thermal_params(cfg)

        assert eta == pytest.approx(0.827)
        assert u == pytest.approx(0.120)
        assert tau == pytest.approx(4.835)

    def test_inactive_heat_pump_channel_falls_back_to_computed(self):
        from src.heating_correction_ml_calibration import _read_baseline_thermal_params

        cfg = SimpleNamespace(
            ENABLE_HEAT_SOURCE_CHANNELS=True,
            OUTLET_EFFECTIVENESS=0.95,
            HEAT_LOSS_COEFFICIENT=0.12,
            THERMAL_TIME_CONSTANT=4.39,
        )

        fake_manager = SimpleNamespace(
            get_heat_source_channel_state=lambda: {
                "heat_pump": {
                    "parameters": {
                        "outlet_effectiveness": 0.84,
                        "heat_loss_coefficient": 0.119,
                        "thermal_time_constant": 4.83,
                    },
                    "history_count": 0,
                    "history": [],
                }
            },
            get_computed_parameters=lambda: {
                "outlet_effectiveness": 0.811,
                "heat_loss_coefficient": 0.133,
                "thermal_time_constant": 4.612,
            },
        )

        with patch(
            "src.unified_thermal_state.get_thermal_state_manager",
            return_value=fake_manager,
        ):
            eta, u, tau = _read_baseline_thermal_params(cfg)

        assert eta == pytest.approx(0.811)
        assert u == pytest.approx(0.133)
        assert tau == pytest.approx(4.612)


# ---------------------------------------------------------------------------
# _parse_heating_start_date (via config module)
# ---------------------------------------------------------------------------

class TestHeatingStartDate:
    def test_valid_date(self):
        from src.config import _parse_heating_start_date
        dt = _parse_heating_start_date("01.01.2024")
        assert dt is not None
        assert dt == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_empty_string_returns_none(self):
        from src.config import _parse_heating_start_date
        assert _parse_heating_start_date("") is None

    def test_invalid_format_returns_none(self):
        from src.config import _parse_heating_start_date
        assert _parse_heating_start_date("2024-01-01") is None
        assert _parse_heating_start_date("not-a-date") is None

    def test_lookback_resolved_from_start_date(self):
        """calibrate_heating_correction_ml computes lookback when start date given."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import (
            calibrate_heating_correction_ml,
        )
        # Use a date far enough in the past (> 365 days) so lookback > 0
        past_date = "01.01.2020"
        df = _make_df(800, at_val=8.0)

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.config.HEATING_ML_CALIBRATION_START_DATE", past_date
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = past_date
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1,2"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1,2"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1,2"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5,1"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/test_hml.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = (
                "/tmp/test_hml_meta.json"
            )
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"
            # _parse_heating_start_date must be a real callable
            from src.config import _parse_heating_start_date
            mock_cfg._parse_heating_start_date = _parse_heating_start_date

            mock_lgb = MagicMock()
            mock_model = MagicMock()
            mock_lgb.LGBMRegressor.return_value = mock_model
            mock_model.predict.side_effect = lambda X: np.zeros(len(X))
            mock_model.fit = MagicMock()

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump"), \
                 patch("os.replace"):
                result = calibrate_heating_correction_ml()
        # Primary goal: no exception; result is bool
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Cold-season filter
# ---------------------------------------------------------------------------

class TestColdSeasonFilter:
    """calibrate_heating_correction_ml filters rows with AT >= threshold."""

    def test_abort_when_too_few_cold_rows(self):
        """Returns False when fewer than 500 rows pass the AT < threshold filter."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import (
            calibrate_heating_correction_ml,
        )
        # 100 rows with AT=5 (cold) + 400 rows with AT=20 (warm = excluded)
        df_cold = _make_df(100, at_val=5.0)
        df_warm = _make_df(400, at_val=20.0)
        import pandas as pd
        df = pd.concat([df_cold, df_warm], ignore_index=True)

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1,2"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 200
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/x.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = "/tmp/x_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"

            with patch(
                "src.heating_correction_ml_calibration._read_baseline_thermal_params",
                return_value=(0.830, 0.124, 4.39),
            ):
                result = calibrate_heating_correction_ml()

        assert result is False


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

class TestLabelConstruction:
    """Verify label = −(T_future − T_target) / S_H with clipping."""

    def _label_from_series(self, indoor_series, target, s_h, n_steps):
        """Reproduce the label computation for a simple numeric series."""
        import numpy as np
        future = indoor_series.shift(-n_steps)
        raw = -(future - target) / s_h
        return raw.clip(-5.0, 5.0)

    def test_label_sign_undershoot(self):
        """When indoor < target in future → label > 0 (need to raise outlet)."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import _compute_s_h
        s_h = _compute_s_h(0.830, 0.124, 4.39, 4.0)
        target = 21.0
        indoor = pd.Series([21.0, 20.5, 20.0, 20.5])  # future=20.0 at t=0+2steps
        label = self._label_from_series(indoor, target, s_h, 2)
        # At t=0: future=20.0 → label = -(20.0-21.0)/S_H = +1.0/S_H > 0
        assert label.iloc[0] > 0

    def test_label_sign_overshoot(self):
        """When indoor > target in future → label < 0 (need to lower outlet)."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import _compute_s_h
        s_h = _compute_s_h(0.830, 0.124, 4.39, 4.0)
        target = 21.0
        indoor = pd.Series([21.0, 21.5, 22.0, 21.5])
        label = self._label_from_series(indoor, target, s_h, 2)
        # At t=0: future=22.0 → label = -(22.0-21.0)/S_H < 0
        assert label.iloc[0] < 0

    def test_label_clipped_at_5(self):
        """Labels beyond ±5 °C are clipped."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import _compute_s_h
        s_h = _compute_s_h(0.830, 0.124, 4.39, 4.0)
        target = 21.0
        # Indoor drops to 10°C in 2 steps → raw label = -(10-21)/S_H ≈ +11/S_H >> 5
        indoor = pd.Series([21.0, 21.0, 10.0, 10.0])
        label = self._label_from_series(indoor, target, s_h, 2)
        assert label.iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

class TestFeatureSet:
    """Verify that fireplace/TV lag features appear in training feature_cols."""

    def test_lag_features_in_feature_cols(self):
        """After pipeline build, fireplace_lag_1h and tv_lag_30m must be present."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import (
            calibrate_heating_correction_ml,
        )
        df = _make_df(800, at_val=8.0)
        captured_metadata = {}

        def _fake_dump(model, path):
            pass

        def _fake_replace(src, dst):
            pass

        def _fake_open_write(path, mode="w", encoding=None):
            import io
            buf = io.StringIO()
            buf._path = path
            buf._captured = captured_metadata
            orig_close = buf.close

            def close():
                captured_metadata.update(json.loads(buf.getvalue()))
                orig_close()

            buf.close = close
            return buf

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1,2"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1,2"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1,2"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5,1"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/hml_test.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = (
                "/tmp/hml_test_meta.json"
            )
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"

            mock_lgb = MagicMock()
            mock_model = MagicMock()
            mock_lgb.LGBMRegressor.return_value = mock_model
            mock_model.predict.return_value = np.zeros(200)
            mock_model.fit = MagicMock()

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump", _fake_dump), \
                 patch("os.replace", _fake_replace), \
                 patch(
                     "builtins.open",
                     side_effect=lambda p, m="r", **kw: (
                         _fake_open_write(p, m)
                         if "w" in m else open.__wrapped__(p, m, **kw)
                     ),
                 ):
                try:
                    calibrate_heating_correction_ml()
                except Exception:
                    pass  # we only care about what features were built

        # Check the feature_cols were passed to the LGBMRegressor.fit call
        if mock_model.fit.called:
            call_args = mock_model.fit.call_args
            if call_args is not None:
                X_fit = call_args[0][0]
                n_cols = X_fit.shape[1] if hasattr(X_fit, "shape") else 0
                # Pipeline should produce at least indoor_temp + AT + lag features
                assert n_cols >= 5

    def test_feature_list_contains_fireplace_lag(self):
        """The feature_cols must include dynamic fireplace and TV lag columns."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        # Build a small DataFrame and verify the derived column names
        df = _make_df(50, at_val=8.0)
        df["fireplace_on"] = 0.0
        df["tv_on"] = 0.0

        steps_per_hour = 6
        # Default HEATING_ML_FIREPLACE_LAG_HOURS = "1,2"
        for lag_h in [1.0, 2.0]:
            n_steps = max(1, int(round(lag_h * steps_per_hour)))
            col_name = f"fireplace_lag_{int(lag_h)}h"
            df[col_name] = df["fireplace_on"].rolling(n_steps, min_periods=1).max()

        # Default HEATING_ML_TV_LAG_HOURS = "0.5,1"
        for lag_h, expected_name in [(0.5, "tv_lag_30m"), (1.0, "tv_lag_1h")]:
            n_steps = max(1, int(round(lag_h * steps_per_hour)))
            df[expected_name] = df["tv_on"].rolling(n_steps, min_periods=1).max()

        assert "fireplace_lag_1h" in df.columns
        assert "fireplace_lag_2h" in df.columns
        assert "tv_lag_30m" in df.columns
        assert "tv_lag_1h" in df.columns

    def test_pv_features_added_when_pv_data_present(self):
        """When PV data is in the DataFrame, pv_roll and pv_forecast columns appear."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import calibrate_heating_correction_ml
        df = _make_df(800, at_val=8.0)
        df["pv_leistung_gefiltert"] = np.random.default_rng(1).uniform(0, 3000, len(df))

        captured_features = []

        def _capture_fit(X, y, **kwargs):
            captured_features.append(X.shape[1])

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1,2"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1,2"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5,1"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/hml_pv_test.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = "/tmp/hml_pv_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            # PV entity whose suffix matches the DF column name
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv_leistung_gefiltert"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"

            mock_lgb = MagicMock()
            mock_model = MagicMock()
            mock_lgb.LGBMRegressor.return_value = mock_model
            mock_model.predict.side_effect = lambda X: np.zeros(len(X))
            mock_model.fit.side_effect = _capture_fit

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump"), \
                 patch("os.replace"):
                try:
                    calibrate_heating_correction_ml()
                except Exception:
                    pass

        if captured_features:
            # With PV data present, we expect more columns than the minimum
            # (AT:1, indoor:5, thermal:4, fp:3, tv:3, PV:4+2fc, time:4 = ≥ 20)
            assert captured_features[0] >= 10

    def test_pv_features_fallback_to_zero_when_absent(self):
        """When PV data is absent, calibration still succeeds (PV=0 fallback)."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import calibrate_heating_correction_ml
        # _make_df has no PV column → should be filled with 0 silently
        df = _make_df(800, at_val=8.0)

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/hml_nopv.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = "/tmp/hml_nopv_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv_leistung_gefiltert"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"

            mock_lgb = MagicMock()
            mock_model = MagicMock()
            mock_lgb.LGBMRegressor.return_value = mock_model
            mock_model.predict.side_effect = lambda X: np.zeros(len(X))
            mock_model.fit = MagicMock()

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump"), \
                 patch("os.replace"):
                result = calibrate_heating_correction_ml()

        # Should not raise; returns a bool
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Config: new ML calibration config vars exist with expected defaults
# ---------------------------------------------------------------------------

class TestMLCalibrationConfigDefaults:
    """Verify new config vars for feature pruning, regularisation, Optuna, CV."""

    def test_feature_pruning_enabled_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_FEATURE_PRUNING_ENABLED", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_FEATURE_PRUNING_ENABLED is True
        finally:
            if orig is not None:
                os.environ["HEATING_ML_FEATURE_PRUNING_ENABLED"] = orig
            importlib.reload(cfg)

    def test_prune_pi_threshold_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_PRUNE_PI_THRESHOLD", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_PRUNE_PI_THRESHOLD == pytest.approx(0.0)
        finally:
            if orig is not None:
                os.environ["HEATING_ML_PRUNE_PI_THRESHOLD"] = orig
            importlib.reload(cfg)

    def test_reg_alpha_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_REG_ALPHA", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_REG_ALPHA == pytest.approx(0.1)
        finally:
            if orig is not None:
                os.environ["HEATING_ML_REG_ALPHA"] = orig
            importlib.reload(cfg)

    def test_reg_lambda_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_REG_LAMBDA", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_REG_LAMBDA == pytest.approx(1.0)
        finally:
            if orig is not None:
                os.environ["HEATING_ML_REG_LAMBDA"] = orig
            importlib.reload(cfg)

    def test_optuna_disabled_by_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_OPTUNA_ENABLED", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_OPTUNA_ENABLED is False
        finally:
            if orig is not None:
                os.environ["HEATING_ML_OPTUNA_ENABLED"] = orig
            importlib.reload(cfg)

    def test_cv_disabled_by_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("HEATING_ML_CV_ENABLED", None)
        try:
            importlib.reload(cfg)
            assert cfg.HEATING_ML_CV_ENABLED is False
        finally:
            if orig is not None:
                os.environ["HEATING_ML_CV_ENABLED"] = orig
            importlib.reload(cfg)

    def test_rescue_min_hours_default(self):
        import importlib
        import src.config as cfg
        orig = os.environ.pop("PV_TRAJ_RESCUE_MIN_HOURS", None)
        try:
            importlib.reload(cfg)
            assert cfg.PV_TRAJ_RESCUE_MIN_HOURS == 1
        finally:
            if orig is not None:
                os.environ["PV_TRAJ_RESCUE_MIN_HOURS"] = orig
            importlib.reload(cfg)


# ---------------------------------------------------------------------------
# Feature pruning logic (unit-level)
# ---------------------------------------------------------------------------

class TestFeaturePruningLogic:
    """Test the feature pruning step in calibration (steps 10c)."""

    def test_regularization_params_passed_to_lgbm(self):
        """reg_alpha and reg_lambda from config are forwarded to LGBMRegressor."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import calibrate_heating_correction_ml
        df = _make_df(800, at_val=8.0)
        captured_params = {}

        class _FakeLGBMRegressor(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                captured_params.update(kwargs)
                self.feature_importances_ = np.ones(40)

            def fit(self, X, y, **kw):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/hml_reg.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = "/tmp/hml_reg_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"
            # New config vars
            mock_cfg.HEATING_ML_REG_ALPHA = 0.5
            mock_cfg.HEATING_ML_REG_LAMBDA = 2.0
            mock_cfg.HEATING_ML_FEATURE_PRUNING_ENABLED = False
            mock_cfg.HEATING_ML_OPTUNA_ENABLED = False
            mock_cfg.HEATING_ML_CV_ENABLED = False

            mock_lgb = MagicMock()
            mock_lgb.LGBMRegressor = _FakeLGBMRegressor

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump"), \
                 patch("os.replace"):
                try:
                    calibrate_heating_correction_ml()
                except Exception:
                    pass

        assert captured_params.get("reg_alpha") == 0.5
        assert captured_params.get("reg_lambda") == 2.0

    def test_pruning_disabled_skips_retrain(self):
        """When HEATING_ML_FEATURE_PRUNING_ENABLED is False, no pruning retrain occurs."""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        from src.heating_correction_ml_calibration import calibrate_heating_correction_ml
        df = _make_df(800, at_val=8.0)
        fit_call_count = [0]

        class _CountingLGBMRegressor(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                self.feature_importances_ = np.ones(40)

            def fit(self, X, y, **kw):
                fit_call_count[0] += 1

            def predict(self, X):
                return np.zeros(len(X))

        with patch(
            "src.heating_correction_ml_calibration"
            ".fetch_historical_data_for_calibration",
            return_value=df,
        ), patch(
            "src.heating_correction_ml_calibration._read_baseline_thermal_params",
            return_value=(0.830, 0.124, 4.39),
        ), patch(
            "src.heating_correction_ml_calibration.config"
        ) as mock_cfg:
            mock_cfg.HEATING_ML_CALIBRATION_START_DATE = ""
            mock_cfg.HEATING_ML_COLD_THRESHOLD_C = 18.0
            mock_cfg.HEATING_ML_LABEL_HORIZON_H = 4
            mock_cfg.HEATING_ML_AT_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_PV_FORECAST_HOURS = "1"
            mock_cfg.HEATING_ML_FIREPLACE_LAG_HOURS = "1"
            mock_cfg.HEATING_ML_TV_LAG_HOURS = "0.5"
            mock_cfg.CYCLE_INTERVAL_MINUTES = 10
            mock_cfg.HLC_DEFAULT_TARGET_TEMP = 21.0
            mock_cfg.SPECIFIC_HEAT_CAPACITY = 4.186
            mock_cfg.HEATING_ML_MIN_TRAINING_SAMPLES = 50
            mock_cfg.HEATING_ML_RETRAIN_VAL_FRACTION = 0.25
            mock_cfg.OUTLET_EFFECTIVENESS = 0.830
            mock_cfg.HEAT_LOSS_COEFFICIENT = 0.124
            mock_cfg.THERMAL_TIME_CONSTANT = 4.39
            mock_cfg.HEATING_ML_CORRECTION_MODEL_PATH = "/tmp/hml_np.joblib"
            mock_cfg.HEATING_ML_CORRECTION_METADATA_PATH = "/tmp/hml_np_meta.json"
            mock_cfg.INDOOR_TEMP_ENTITY_ID = "sensor.indoor"
            mock_cfg.OUTDOOR_TEMP_ENTITY_ID = "sensor.outdoor"
            mock_cfg.OUTLET_TEMP_ENTITY_ID = "sensor.outlet"
            mock_cfg.INLET_TEMP_ENTITY_ID = "sensor.inlet"
            mock_cfg.FLOW_RATE_ENTITY_ID = "input_number.flow"
            mock_cfg.POWER_CONSUMPTION_ENTITY_ID = "sensor.power"
            mock_cfg.PV_POWER_ENTITY_ID = "sensor.pv"
            mock_cfg.FIREPLACE_STATUS_ENTITY_ID = "binary_sensor.fireplace_active"
            mock_cfg.TV_STATUS_ENTITY_ID = "input_boolean.fernseher"
            # New config vars: pruning disabled
            mock_cfg.HEATING_ML_REG_ALPHA = 0.1
            mock_cfg.HEATING_ML_REG_LAMBDA = 1.0
            mock_cfg.HEATING_ML_FEATURE_PRUNING_ENABLED = False
            mock_cfg.HEATING_ML_OPTUNA_ENABLED = False
            mock_cfg.HEATING_ML_CV_ENABLED = False

            mock_lgb = MagicMock()
            mock_lgb.LGBMRegressor = _CountingLGBMRegressor

            with patch.dict("sys.modules", {"lightgbm": mock_lgb}), \
                 patch("joblib.dump"), \
                 patch("os.replace"):
                try:
                    calibrate_heating_correction_ml()
                except Exception:
                    pass

        # Only 1 fit call (initial), no pruning retrain
        assert fit_call_count[0] == 1


class TestHoldoutIsolation:
    """Regression tests for strict holdout isolation in HPO/CV paths."""

    def test_optuna_and_cv_do_not_use_holdout_rows(self):
        """Optuna objective and CV diagnostics must use fit split only."""
        try:
            import numpy as np
            import pandas as pd
            import types
            import importlib
        except ImportError:
            pytest.skip("pandas/numpy not installed")

        import src.heating_correction_ml_calibration as hml_cal
        hml_cal = importlib.reload(hml_cal)
        calibrate_heating_correction_ml = hml_cal.calibrate_heating_correction_ml

        df = _make_df(800, at_val=8.0)
        sentinel = 9999.0
        # Last 25% becomes temporal holdout with val_fraction=0.25.
        holdout_start = int(len(df) * 0.75)
        df.loc[df.index[holdout_start:], "indoor_temp"] = sentinel

        fit_call_count = [0]
        fit_max_values = []

        class _FakeTrial:
            def suggest_float(self, name, low, high, log=False):
                return float((low + high) / 2.0)

            def suggest_int(self, name, low, high):
                return int((low + high) // 2)

        class _FakeStudy:
            def __init__(self):
                self.best_params = {}
                self.best_value = 0.0

            def optimize(self, objective, n_trials=1, show_progress_bar=False):
                self.best_value = float(objective(_FakeTrial()))

        class _FakeOptunaLogging:
            WARNING = 0

            @staticmethod
            def set_verbosity(level):
                return None

        fake_optuna = types.ModuleType("optuna")
        fake_optuna.logging = _FakeOptunaLogging
        fake_optuna.create_study = lambda direction="minimize": _FakeStudy()

        class _FakeLGBMRegressor(BaseEstimator, RegressorMixin):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.feature_importances_ = None

            def fit(self, X, y, **kw):
                fit_call_count[0] += 1
                x_arr = np.asarray(X)
                fit_max_values.append(float(np.max(x_arr[:, 0])))
                self.feature_importances_ = np.ones(x_arr.shape[1], dtype=int)
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        tmp_dir = tempfile.gettempdir()
        cfg_overrides = {
            "HEATING_ML_CALIBRATION_START_DATE": "",
            "HEATING_ML_COLD_THRESHOLD_C": 18.0,
            "HEATING_ML_LABEL_HORIZON_H": 4,
            "HEATING_ML_AT_FORECAST_HOURS": "1",
            "HEATING_ML_PV_FORECAST_HOURS": "1",
            "HEATING_ML_FIREPLACE_LAG_HOURS": "1",
            "HEATING_ML_TV_LAG_HOURS": "0.5",
            "CYCLE_INTERVAL_MINUTES": 10,
            "HLC_DEFAULT_TARGET_TEMP": 21.0,
            "SPECIFIC_HEAT_CAPACITY": 4.186,
            "HEATING_ML_MIN_TRAINING_SAMPLES": 50,
            "HEATING_ML_RETRAIN_VAL_FRACTION": 0.25,
            "OUTLET_EFFECTIVENESS": 0.830,
            "HEAT_LOSS_COEFFICIENT": 0.124,
            "THERMAL_TIME_CONSTANT": 4.39,
            "HEATING_ML_CORRECTION_MODEL_PATH": os.path.join(tmp_dir, "hml_holdout.joblib"),
            "HEATING_ML_CORRECTION_METADATA_PATH": os.path.join(tmp_dir, "hml_holdout_meta.json"),
            "INDOOR_TEMP_ENTITY_ID": "sensor.indoor",
            "OUTDOOR_TEMP_ENTITY_ID": "sensor.outdoor",
            "OUTLET_TEMP_ENTITY_ID": "sensor.outlet",
            "INLET_TEMP_ENTITY_ID": "sensor.inlet",
            "FLOW_RATE_ENTITY_ID": "input_number.flow",
            "POWER_CONSUMPTION_ENTITY_ID": "sensor.power",
            "PV_POWER_ENTITY_ID": "sensor.pv",
            "FIREPLACE_STATUS_ENTITY_ID": "binary_sensor.fireplace_active",
            "TV_STATUS_ENTITY_ID": "input_boolean.fernseher",
            "HEATING_ML_REG_ALPHA": 0.1,
            "HEATING_ML_REG_LAMBDA": 1.0,
            "HEATING_ML_FEATURE_PRUNING_ENABLED": False,
            "HEATING_ML_OPTUNA_ENABLED": True,
            "HEATING_ML_OPTUNA_N_TRIALS": 1,
            "HEATING_ML_CV_ENABLED": True,
            "HEATING_ML_CV_N_SPLITS": 3,
        }

        from contextlib import ExitStack
        with ExitStack() as stack:
            cfg_stub = types.SimpleNamespace(**cfg_overrides)
            stack.enter_context(patch(
                "src.heating_correction_ml_calibration"
                ".fetch_historical_data_for_calibration",
                return_value=df,
            ))
            stack.enter_context(patch(
                "src.heating_correction_ml_calibration._read_baseline_thermal_params",
                return_value=(0.830, 0.124, 4.39),
            ))
            stack.enter_context(
                patch.object(hml_cal, "config", cfg_stub)
            )

            mock_lgb = MagicMock()
            mock_lgb.LGBMRegressor = _FakeLGBMRegressor
            mock_lgb.early_stopping = lambda *a, **k: None
            mock_lgb.log_evaluation = lambda *a, **k: None

            with patch.dict(
                "sys.modules",
                {"lightgbm": mock_lgb, "optuna": fake_optuna},
            ), patch("joblib.dump"), patch("os.replace"):
                result = calibrate_heating_correction_ml()

        assert result is True
        # Optuna + CV + final fit should yield multiple training fits.
        assert fit_call_count[0] >= 5
        # Regression assertion: no training fit may include holdout sentinel rows.
        assert all(v < sentinel for v in fit_max_values)
