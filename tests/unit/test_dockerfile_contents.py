from pathlib import Path


def test_dockerfile_copies_dashboard_config_bundle():
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "COPY ml_heating_underfloor/ /app/ml_heating_underfloor/" in content
