import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from model_utils import (
    build_rf_features,
    build_mlp_features,
    build_ae_features,
)


def test_rf_features():
    result = build_rf_features(
        "M",
        298.1,
        308.6,
        1500,
        40,
        100,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    expected_columns = [
        "type",
        "air_temp_k",
        "process_temp_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "temp_diff_k",
        "power_kw",
        "temp_ratio",
        "torque_per_rpm",
        "temp_x_speed",
        "torque_x_wear",
    ]

    assert list(result.columns) == expected_columns


def test_mlp_features():
    result = build_mlp_features(
        "M",
        298.1,
        308.6,
        1500,
        40,
        100,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "wear_rate" in result.columns
    assert "torque_toolwear_interaction" in result.columns


def test_autoencoder_features():
    result = build_ae_features(
        298.1,
        308.6,
        1500,
        40,
        100,
    )

    assert result.shape == (1, 11)


def test_feature_calculations():
    result = build_rf_features(
        "M",
        298.1,
        308.6,
        1500,
        40,
        100,
    )

    assert result["temp_diff_k"].iloc[0] == 10.5
    assert result["power_kw"].iloc[0] > 0
    assert result["torque_per_rpm"].iloc[0] > 0