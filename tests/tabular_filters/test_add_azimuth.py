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


def test_add_azimuth():
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "spacecraft_latitude": [-11.0, 1.0, 11.0],
            "spacecraft_longitude": [1.0, 91.0, 271.0],
        }
    )
    add_azimuth = create_filter("add_azimuth")
    result = add_azimuth(df.copy())
    new_col = "azimuth"

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + (new_col,)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "spacecraft_latitude", "spacecraft_longitude"]].equals(
        df[["latitude", "longitude", "spacecraft_latitude", "spacecraft_longitude"]]
    )
    expected_azimuth = np.array([135.57378316, 44.99563646, 44.42621684])
    assert np.allclose(result[new_col].to_numpy(), expected_azimuth)


def test_add_azimuth_with_config():
    config = {
        "azimuth": "az",
        "spacecraft_latitude": "sc_lat",
        "spacecraft_longitude": "sc_lon",
    }
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "sc_lat": [-11.0, 1.0, 11.0],
            "sc_lon": [1.0, 91.0, 271.0],
        }
    )
    add_azimuth = create_filter("add_azimuth", **config)
    result = add_azimuth(df.copy())
    new_col = config["azimuth"]

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + (new_col,)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude", "sc_lat", "sc_lon"]].equals(
        df[["latitude", "longitude", "sc_lat", "sc_lon"]]
    )
    expected_azimuth = np.array([135.57378316, 44.99563646, 44.42621684])
    assert np.allclose(result[new_col].to_numpy(), expected_azimuth)


def test_add_azimuth_missing_column():
    df = pd.DataFrame(
        {
            "latitude": [-10.0, 0.0, 10.0],
            "longitude": [0.0, 90.0, 270.0],
            "spacecraft_latitude": [-11.0, 1.0, 11.0],
            # missing spacecraft_longitude
        }
    )
    add_azimuth = create_filter("add_azimuth")
    with pytest.raises(ValueError):
        _ = add_azimuth(df.copy())
