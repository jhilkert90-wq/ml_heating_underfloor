"""
Shared settings metadata for the Streamlit dashboard.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "ml_heating_underfloor" / "config.yaml"
_EN_TRANSLATIONS_PATH = (
    _REPO_ROOT / "ml_heating_underfloor" / "translations" / "en.yaml"
)


@dataclass(frozen=True)
class GroupDefinition:
    slug: str
    prefix: str
    title: str
    expanded: bool = False


GROUP_DEFINITIONS = OrderedDict(
    (
        ("core", GroupDefinition("core", "[Core]", "Core Entities")),
        ("blocking", GroupDefinition("blocking", "[Blocking]", "Blocking Detection")),
        ("safety", GroupDefinition("safety", "[Safety]", "Safety Limits")),
        ("learning", GroupDefinition("learning", "[Learning]", "Learning Parameters")),
        ("thermal_model", GroupDefinition("thermal_model", "[Thermal]", "Thermal Model")),
        ("heat_sources", GroupDefinition("heat_sources", "[Heat Sources]", "External Heat Sources")),
        ("trajectory", GroupDefinition("trajectory", "[Trajectory]", "Trajectory & Prediction")),
        ("functions", GroupDefinition("functions", "[Functions]", "Features (On/Off)")),
        ("cooling", GroupDefinition("cooling", "[Cooling]", "Cooling Mode")),
        ("pre_cooling", GroupDefinition("pre_cooling", "[Pre-Cooling]", "Pre-Cooling")),
        ("ml_pre_cooling", GroupDefinition("ml_pre_cooling", "[ML Pre-Cooling]", "ML Pre-Cooling Model")),
        ("ml_heating", GroupDefinition("ml_heating", "[ML Heating]", "ML Heating Correction")),
        ("hlc", GroupDefinition("hlc", "[HLC]", "HLC Calibration")),
        ("price_pv", GroupDefinition("price_pv", "[Price/PV]", "Price & PV Optimization")),
        ("shadow", GroupDefinition("shadow", "[Shadow]", "Shadow Mode")),
        ("outlet", GroupDefinition("outlet", "[Outlet]", "Outlet Smoothing")),
        ("influxdb", GroupDefinition("influxdb", "[InfluxDB]", "InfluxDB")),
        ("model", GroupDefinition("model", "[Model]", "Model Management")),
        ("dashboard", GroupDefinition("dashboard", "[Dashboard]", "Dashboard")),
        ("dev", GroupDefinition("dev", "[Dev]", "Development & Debug")),
        ("advanced", GroupDefinition("advanced", "[Advanced]", "Advanced Settings")),
    )
)


_SECTION_TO_GROUP = {
    "Core Entity Configuration": "core",
    "External Heat Sources": "heat_sources",
    "Blocking Detection": "blocking",
    "Blocking Recovery": "blocking",
    "ML Learning Parameters": "learning",
    "Adaptive Learning Parameters": "learning",
    "Hybrid Learning Strategy": "learning",
    "Learning History Sizes": "learning",
    "Indoor Trend Protection": "learning",
    "Safety Configuration": "safety",
    "Cooling Mode Configuration": "cooling",
    "Pre-Cooling (Predictive Overheating Prevention)": "pre_cooling",
    "ML-Based Pre-Cooling Model (LightGBM Overheating Classifier)": "ml_pre_cooling",
    "ML-Based Heating Correction (LightGBM Regressor)": "ml_heating",
    "Heating Correction Mode": "ml_heating",
    "Thermal Equilibrium Model Parameters": "thermal_model",
    "Heat Source Channel Architecture": "functions",
    "External Heat Source Weights": "heat_sources",
    "Trajectory Tuning": "trajectory",
    "Trajectory Prediction": "functions",
    "Advanced Learning Features": "functions",
    "Prediction Metrics Tracking": "functions",
    "Delta Temperature Forecast Calibration": "functions",
    "Setpoint Stability": "trajectory",
    "Forecast-Driven Trajectory Scaling": "trajectory",
    "Historical Calibration System": "hlc",
    "Thermal Power Gate Thresholds": "hlc",
    "HLC Validation Gates": "hlc",
    "PV-Triggered HLC Session Learner": "hlc",
    "Historical HLC Calibration": "hlc",
    "Electricity Price Optimization (Tibber)": "price_pv",
    "PV Surplus Optimization": "price_pv",
    "Shadow Mode Configuration": "shadow",
    "Outlet Smoothing": "outlet",
    "InfluxDB Configuration": "influxdb",
    "Model Management": "model",
    "Dashboard Configuration": "dashboard",
    "Development Settings": "dev",
}


@dataclass(frozen=True)
class FieldMetadata:
    key: str
    label: str
    description: str
    schema: str
    default: Any
    group: GroupDefinition
    widget_type: str
    options: tuple[Any, ...] = ()
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    is_secret: bool = False


@dataclass(frozen=True)
class SettingsMetadata:
    fields: dict[str, FieldMetadata]
    field_order: tuple[str, ...]
    defaults: dict[str, Any]
    groups: OrderedDict[str, GroupDefinition]


def _extract_option_groups() -> dict[str, str]:
    groups: dict[str, str] = {}
    current_group = "advanced"
    in_options = False
    for raw_line in _CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped == "options:":
            in_options = True
            continue
        if stripped == "schema:":
            break
        if not in_options:
            continue

        section_match = re.match(r"# === (.+?) ===$", stripped)
        if section_match:
            current_group = _SECTION_TO_GROUP.get(section_match.group(1), "advanced")
            continue

        key_match = re.match(r"^  ([a-z0-9_]+):", raw_line)
        if key_match:
            groups[key_match.group(1)] = current_group
    return groups


def _parse_schema(key: str, raw_schema: str) -> dict[str, Any]:
    if raw_schema == "bool":
        return {"widget_type": "bool"}
    if raw_schema in {"str", "str?"}:
        return {"widget_type": "text"}
    if raw_schema.startswith("list(") and raw_schema.endswith(")"):
        options = tuple(raw_schema[5:-1].split("|"))
        return {"widget_type": "select", "options": options}
    num_match = re.match(r"^(int|float)\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)$", raw_schema)
    if num_match:
        num_type = num_match.group(1)
        min_value = float(num_match.group(2))
        max_value = float(num_match.group(3))
        if num_type == "int":
            min_value = int(min_value)
            max_value = int(max_value)
            step = 1
        else:
            span = abs(max_value - min_value)
            step = 0.1 if span <= 10 else 1.0
        return {
            "widget_type": num_type,
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
        }
    return {"widget_type": "text"}


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return "token" in lowered or lowered.endswith("_key") or "password" in lowered


@lru_cache(maxsize=1)
def load_settings_metadata() -> SettingsMetadata:
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    en_translations = yaml.safe_load(_EN_TRANSLATIONS_PATH.read_text(encoding="utf-8"))

    options = config["options"]
    schema = config["schema"]
    en_config = en_translations["configuration"]
    option_groups = _extract_option_groups()

    fields: dict[str, FieldMetadata] = {}
    field_order: list[str] = []

    for key, default in options.items():
        schema_meta = _parse_schema(key, schema[key])
        group_slug = option_groups.get(key, "advanced")
        group = GROUP_DEFINITIONS[group_slug]
        en_entry = en_config.get(key, {})
        fields[key] = FieldMetadata(
            key=key,
            label=en_entry.get("name", key),
            description=en_entry.get("description", ""),
            schema=schema[key],
            default=default,
            group=group,
            widget_type=schema_meta["widget_type"],
            options=schema_meta.get("options", ()),
            min_value=schema_meta.get("min_value"),
            max_value=schema_meta.get("max_value"),
            step=schema_meta.get("step"),
            is_secret=_is_secret_key(key),
        )
        field_order.append(key)

    return SettingsMetadata(
        fields=fields,
        field_order=tuple(field_order),
        defaults=dict(options),
        groups=GROUP_DEFINITIONS,
    )
