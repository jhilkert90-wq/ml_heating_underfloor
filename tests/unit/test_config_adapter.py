from config_adapter import convert_addon_to_env


def test_convert_addon_to_env_maps_learning_threshold_options():
    env = convert_addon_to_env(
        {
            "learning_dead_zone": 0.02,
            "pv_learning_threshold": 75,
            "recent_errors_window": 7,
        }
    )

    assert env["LEARNING_DEAD_ZONE"] == "0.02"
    assert env["PV_LEARNING_THRESHOLD"] == "75"
    assert env["RECENT_ERRORS_WINDOW"] == "7"
