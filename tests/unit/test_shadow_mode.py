import pytest

from src import shadow_mode


def test_resolve_shadow_mode_separates_deployment_and_effective_cycle(monkeypatch):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", False)

    decision = shadow_mode.resolve_shadow_mode(ml_heating_enabled=False)

    assert decision.shadow_deployment is False
    assert decision.effective_shadow_mode is True


def test_get_shadow_output_entity_id_preserves_domain(monkeypatch):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", True)

    result = shadow_mode.get_shadow_output_entity_id(
        "sensor.ml_vorlauftemperatur"
    )

    assert result == "sensor.ml_vorlauftemperatur_shadow"


def test_get_shadow_output_entity_id_is_idempotent(monkeypatch):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", True)

    result = shadow_mode.get_shadow_output_entity_id(
        "sensor.ml_vorlauftemperatur_shadow"
    )

    assert result == "sensor.ml_vorlauftemperatur_shadow"


def test_get_effective_influx_features_bucket_suffixes_only_in_shadow(monkeypatch):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", True)

    result = shadow_mode.get_effective_influx_features_bucket(
        "ml_heating_features"
    )

    assert result == "ml_heating_features_shadow"


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        (
            "/opt/ml_heating/unified_thermal_state.json",
            "/opt/ml_heating/unified_thermal_state_shadow.json",
        ),
        (
            r"C:\ml_heating\unified_thermal_state.json",
            r"C:\ml_heating\unified_thermal_state_shadow.json",
        ),
        (
            "unified_thermal_state",
            "unified_thermal_state_shadow",
        ),
    ],
)
def test_get_shadow_output_file_path_preserves_extension_and_directory(
    monkeypatch, file_path, expected
):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", True)

    result = shadow_mode.get_shadow_output_file_path(file_path)

    assert result == expected


def test_get_effective_unified_state_file_uses_base_config(monkeypatch):
    monkeypatch.setattr(shadow_mode.config, "SHADOW_MODE", True)
    monkeypatch.setattr(
        shadow_mode.config,
        "UNIFIED_STATE_FILE",
        "/opt/ml_heating/unified_thermal_state.json",
    )

    result = shadow_mode.get_effective_unified_state_file()

    assert result == "/opt/ml_heating/unified_thermal_state_shadow.json"