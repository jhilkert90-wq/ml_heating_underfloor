"""Shadow-mode decision and naming helpers.

Separate static shadow deployment identity from per-cycle effective shadow
behavior so control, output naming, and state isolation can share one
decision source.
"""

from dataclasses import dataclass
from typing import Optional

from . import config


SHADOW_SUFFIX = "_shadow"


@dataclass(frozen=True)
class ShadowModeDecision:
    """Resolved shadow-mode state for the current deployment or cycle."""

    shadow_deployment: bool
    effective_shadow_mode: bool

    @property
    def should_publish_live_outputs(self) -> bool:
        """Return whether live Home Assistant outputs should be updated."""
        return not self.effective_shadow_mode

    @property
    def should_control_heating(self) -> bool:
        """Return whether the app should control the live target entity."""
        return not self.effective_shadow_mode

    @property
    def should_publish_output_entities(self) -> bool:
        """Return whether this cycle should publish HA output entities."""
        return self.shadow_deployment or not self.effective_shadow_mode


def _use_shadow_deployment(shadow_deployment: Optional[bool] = None) -> bool:
    """Return whether shadow deployment identity is active."""
    if shadow_deployment is not None:
        return bool(shadow_deployment)
    return resolve_shadow_mode().shadow_deployment


def _append_shadow_suffix(value: str) -> str:
    """Append the deployment suffix once without double-applying it."""
    if not value or value.endswith(SHADOW_SUFFIX):
        return value
    return f"{value}{SHADOW_SUFFIX}"


def _strip_shadow_suffix(value: str) -> str:
    """Remove the deployment suffix once if it is present."""
    if value.endswith(SHADOW_SUFFIX):
        return value[: -len(SHADOW_SUFFIX)]
    return value


def get_shadow_output_entity_id(
    entity_id: str,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the effective output entity id for this deployment."""
    if not _use_shadow_deployment(shadow_deployment):
        return entity_id

    domain, separator, object_id = entity_id.partition(".")
    if not separator:
        return _append_shadow_suffix(entity_id)
    return f"{domain}.{_append_shadow_suffix(object_id)}"


def get_base_output_entity_id(entity_id: str) -> str:
    """Return the unsuffixed form of an output entity id."""
    domain, separator, object_id = entity_id.partition(".")
    if not separator:
        return _strip_shadow_suffix(entity_id)
    return f"{domain}.{_strip_shadow_suffix(object_id)}"


def get_shadow_output_bucket_name(
    bucket_name: str,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the effective output bucket name for this deployment."""
    if not _use_shadow_deployment(shadow_deployment):
        return bucket_name
    return _append_shadow_suffix(bucket_name)


def get_shadow_output_file_path(
    file_path: str,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the effective output file path for this deployment."""
    if not _use_shadow_deployment(shadow_deployment):
        return file_path

    last_separator = max(file_path.rfind("/"), file_path.rfind("\\"))
    directory = file_path[: last_separator + 1] if last_separator >= 0 else ""
    filename = file_path[last_separator + 1 :]

    extension_index = filename.find(".", 1)
    if extension_index == -1:
        suffixed_filename = _append_shadow_suffix(filename)
    else:
        basename = filename[:extension_index]
        extension = filename[extension_index:]
        suffixed_filename = f"{_append_shadow_suffix(basename)}{extension}"

    return f"{directory}{suffixed_filename}"


def get_effective_unified_state_file(
    state_file: Optional[str] = None,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the unified state file path for the current deployment."""
    return get_shadow_output_file_path(
        state_file or config.UNIFIED_STATE_FILE,
        shadow_deployment=shadow_deployment,
    )


def get_effective_cooling_state_file(
    state_file: Optional[str] = None,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the cooling state file path for the current deployment."""
    return get_shadow_output_file_path(
        state_file or config.UNIFIED_STATE_FILE_COOLING,
        shadow_deployment=shadow_deployment,
    )


def get_effective_influx_features_bucket(
    bucket_name: Optional[str] = None,
    *,
    shadow_deployment: Optional[bool] = None,
) -> str:
    """Return the features bucket name for the current deployment."""
    return get_shadow_output_bucket_name(
        bucket_name or config.INFLUX_FEATURES_BUCKET,
        shadow_deployment=shadow_deployment,
    )


def resolve_shadow_mode(
    *,
    ml_heating_enabled: Optional[bool] = None,
    effective_shadow_mode: Optional[bool] = None,
) -> ShadowModeDecision:
    """Resolve static deployment identity and effective shadow behavior."""
    shadow_deployment = bool(config.SHADOW_MODE)

    if effective_shadow_mode is not None:
        return ShadowModeDecision(
            shadow_deployment=shadow_deployment,
            effective_shadow_mode=bool(effective_shadow_mode),
        )

    if ml_heating_enabled is None:
        return ShadowModeDecision(
            shadow_deployment=shadow_deployment,
            effective_shadow_mode=shadow_deployment,
        )

    return ShadowModeDecision(
        shadow_deployment=shadow_deployment,
        effective_shadow_mode=shadow_deployment or not bool(ml_heating_enabled),
    )