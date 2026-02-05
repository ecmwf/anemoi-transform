# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter


def test_add_msg_angles_azimuth_default_config():
    config = {"angle": "azimuth"}
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "satellite_id": [55, 56, 57],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles", **config)
    result = add_msg_angles(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("azimuth",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "satellite_id", "datetime"]].equals(
        df[["latitude", "longitude", "satellite_id", "datetime"]]
    )
    expected_azimuth = np.array([78.703325, 0.0, 90.0])
    assert np.allclose(result["azimuth"].to_numpy(), expected_azimuth)


def test_add_msg_angles_azimuth_with_config():
    config = {"satellite_id": "satid", "azimuth": "a", "angle": "azimuth"}
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "satid": [55, 56, 57],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles", **config)
    result = add_msg_angles(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("a",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "satid", "datetime"]].equals(
        df[["latitude", "longitude", "satid", "datetime"]]
    )
    expected_azimuth = np.array([78.703325, 0.0, 90.0])
    assert np.allclose(result["a"].to_numpy(), expected_azimuth)


def test_add_msg_angles_zenith_default_config():
    config = {"angle": "zenith"}
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "satellite_id": [55, 56, 57],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles", **config)
    result = add_msg_angles(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("zenith",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "satellite_id", "datetime"]].equals(
        df[["latitude", "longitude", "satellite_id", "datetime"]]
    )
    expected_zenith = np.array([48.49626885, 51.82994258, 98.60173361])
    assert np.allclose(result["zenith"].to_numpy(), expected_zenith)


def test_add_msg_angles_zenith_with_config():
    config = {"satellite_id": "satid", "zenith": "z", "angle": "zenith"}
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "satid": [55, 56, 57],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles", **config)
    result = add_msg_angles(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("z",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "satid", "datetime"]].equals(
        df[["latitude", "longitude", "satid", "datetime"]]
    )
    expected_zenith = np.array([48.49626885, 51.82994258, 98.60173361])
    assert np.allclose(result["z"].to_numpy(), expected_zenith)


def test_add_msg_angles_both_default_config():
    config = {"angle": "both"}
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "satellite_id": [55, 56, 57],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles", **config)
    result = add_msg_angles(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(list(df.columns) + ["azimuth", "zenith"])
    assert result.shape == (len(df), len(df.columns) + 2)

    assert result[["latitude", "longitude", "satellite_id", "datetime"]].equals(
        df[["latitude", "longitude", "satellite_id", "datetime"]]
    )
    expected_azimuth = np.array([78.703325, 0.0, 90.0])
    assert np.allclose(result["azimuth"].to_numpy(), expected_azimuth)
    expected_zenith = np.array([48.49626885, 51.82994258, 98.60173361])
    assert np.allclose(result["zenith"].to_numpy(), expected_zenith)


def test_add_msg_angles_missing_satellite_id():
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "datetime": pd.date_range("2025-01-01", periods=3, freq="1H"),
        }
    )
    add_msg_angles = create_filter("add_msg_angles")
    with pytest.raises(ValueError):
        _ = add_msg_angles(df.copy())


def test_add_msg_angles_invald_angle():
    config = {"angle": "invalid_angle"}
    with pytest.raises(ValueError):
        _ = create_filter("add_msg_angles", **config)
