"""Image smoke tests for the ML Heating Underfloor Docker image.

These tests pull and inspect the built image, then run quick ``docker run``
checks to verify:
- Required OCI / Home Assistant add-on labels are present
- ML Python dependencies import cleanly
- Core source modules are importable
- The config adapter is usable
- Required files exist inside the container

Run locally (after building the image) by setting the IMAGE env var::

    IMAGE=ghcr.io/jhilkert90-wq/ml_heating_underfloor-amd64:0.2.44 \\
        pytest tests/integration/test_image_smoke.py -v

The tests are marked ``integration`` and are *not* collected by the default
pytest run (``pytest tests/unit/``).  They require Docker to be available on
the host and the target image to be accessible.
"""

from __future__ import annotations

import json
import os
import subprocess
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE = "ghcr.io/jhilkert90-wq/ml_heating_underfloor-amd64:latest"


def _image() -> str:
    return os.environ.get("IMAGE", _DEFAULT_IMAGE)


def _docker_run(*args: str, entrypoint: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "run", "--rm"]
    if entrypoint is not None:
        cmd += ["--entrypoint", entrypoint]
    cmd.append(_image())
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True)


def _docker_inspect() -> dict:
    result = subprocess.run(
        ["docker", "inspect", _image()],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return data[0] if data else {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def pull_image():
    """Pull the image once for the entire module."""
    subprocess.run(["docker", "pull", _image()], check=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_oci_labels():
    """OCI and Home Assistant add-on labels must be set correctly."""
    info = _docker_inspect()
    labels: dict[str, str] = info.get("Config", {}).get("Labels", {})

    assert labels.get("io.hass.type") == "addon", (
        f"Expected io.hass.type='addon', got {labels.get('io.hass.type')!r}"
    )
    assert labels.get("io.hass.version"), "io.hass.version label is missing or empty"
    assert labels.get("org.opencontainers.image.version"), (
        "org.opencontainers.image.version label is missing or empty"
    )
    # Both version labels should agree
    assert labels["io.hass.version"] == labels["org.opencontainers.image.version"], (
        f"Version label mismatch: io.hass.version={labels['io.hass.version']!r} vs "
        f"org.opencontainers.image.version={labels['org.opencontainers.image.version']!r}"
    )


@pytest.mark.integration
def test_ml_dependencies():
    """Key ML Python packages must be importable inside the container."""
    result = _docker_run(
        "-c",
        "import lightgbm, pandas, numpy, sklearn, streamlit; print('ML deps OK')",
        entrypoint="python3",
    )
    assert result.returncode == 0, f"ML dependency import failed:\n{result.stderr}"
    assert "ML deps OK" in result.stdout


@pytest.mark.integration
def test_core_modules():
    """Core application modules must be importable from /app."""
    script = (
        "import sys; sys.path.insert(0, '/app'); "
        "from src.config import CYCLE_INTERVAL_MINUTES; "
        "from src.thermal_equilibrium_model import ThermalEquilibriumModel; "
        "from src.model_wrapper import EnhancedModelWrapper; "
        "print('Core modules OK')"
    )
    result = _docker_run("-c", script, entrypoint="python3")
    assert result.returncode == 0, f"Core module import failed:\n{result.stderr}"
    assert "Core modules OK" in result.stdout


@pytest.mark.integration
def test_config_adapter():
    """config_adapter.convert_addon_to_env must be importable and callable."""
    script = (
        "import sys; sys.path.insert(0, '/app'); "
        "from config_adapter import convert_addon_to_env; "
        "assert callable(convert_addon_to_env); "
        "print('Config adapter OK')"
    )
    result = _docker_run("-c", script, entrypoint="python3")
    assert result.returncode == 0, f"Config adapter import failed:\n{result.stderr}"
    assert "Config adapter OK" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    "path",
    [
        "/app/run.sh",
        "/app/validate_container.py",
        "/app/src/main.py",
        "/app/config_adapter.py",
    ],
)
def test_required_files_exist(path: str):
    """Required files must be present inside the container."""
    result = _docker_run("-f", path, entrypoint="test")
    assert result.returncode == 0, f"Required file not found inside image: {path}"
