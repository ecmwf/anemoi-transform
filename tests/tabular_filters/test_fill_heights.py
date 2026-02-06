# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from unittest import mock

import numpy as np
import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter
from tests.utils import mock_zarr_dataset


def test_fill_heights_defaults():
    config = {
        "orography_file": "/path/to/orography.zarr",
    }
    df = pd.DataFrame(
        {
            "latitude": [-45.0, -45.0, 45.0, 45.0],
            "longitude": [0.0, 180.0, 0.0, 180.0],
            "stalt": [np.nan, 9999.0, 3.0, np.nan],
        }
    )

    with mock.patch("anemoi.transform.filters.tabular.fill_heights.zarr.open") as open_zarr:
        zarr_dataset = mock_zarr_dataset(
            {
                "latitude": [-45.0, 45.0],
                "longitude": [0.0, 180.0],
                "z": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        open_zarr.return_value = zarr_dataset

        fill_heights = create_filter("fill_orography", **config)
        result = fill_heights(df.copy())
        open_zarr.assert_called_once_with(config["orography_file"], mode="r")

        assert isinstance(result, pd.DataFrame)
        assert tuple(result.columns) == tuple(df.columns)
        assert result.shape == df.shape

        assert result[["latitude", "longitude"]].equals(df[["latitude", "longitude"]])
        # limiting case where all z values are in the dataframe
        # Result should be flattened z values corresponding to each (latitude, longitude) pair
        assert np.allclose(result["stalt"], [1.0, 2.0, 3.0, 4.0])


def test_fill_heights_station_altitude():
    config = {"orography_file": "/path/to/orography.zarr", "station_altitude": "my_station_altitude"}
    df = pd.DataFrame(
        {
            "latitude": [-45.0, -45.0, 45.0, 45.0],
            "longitude": [0.0, 180.0, 0.0, 180.0],
            "my_station_altitude": [np.nan, 9999.0, 3.0, np.nan],
        }
    )

    with mock.patch("anemoi.transform.filters.tabular.fill_heights.zarr.open") as open_zarr:
        zarr_dataset = mock_zarr_dataset(
            {
                "latitude": [-45.0, 45.0],
                "longitude": [0.0, 180.0],
                "z": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        open_zarr.return_value = zarr_dataset

        fill_heights = create_filter("fill_orography", **config)
        result = fill_heights(df.copy())
        open_zarr.assert_called_once_with(config["orography_file"], mode="r")

        assert isinstance(result, pd.DataFrame)
        assert tuple(result.columns) == tuple(df.columns)
        assert result.shape == df.shape

        assert result[["latitude", "longitude"]].equals(df[["latitude", "longitude"]])
        # limiting case where all z values are in the dataframe
        # Result should be flattened z values corresponding to each (latitude, longitude) pair
        assert np.allclose(result["my_station_altitude"], [1.0, 2.0, 3.0, 4.0])


def test_fill_heights_orog_file_varnames():
    config = {
        "orography_file": "/path/to/orography.zarr",
        "orography_altitude": "orog",
        "orography_latitude": "lat",
        "orography_longitude": "lon",
    }
    df = pd.DataFrame(
        {
            "latitude": [-45.0, -45.0, 45.0, 45.0],
            "longitude": [0.0, 180.0, 0.0, 180.0],
            "stalt": [np.nan, 9999.0, 3.0, np.nan],
        }
    )

    with mock.patch("anemoi.transform.filters.tabular.fill_heights.zarr.open") as open_zarr:
        zarr_dataset = mock_zarr_dataset(
            {
                "lat": [-45.0, 45.0],
                "lon": [0.0, 180.0],
                "orog": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        open_zarr.return_value = zarr_dataset

        fill_heights = create_filter("fill_orography", **config)
        result = fill_heights(df.copy())
        open_zarr.assert_called_once_with(config["orography_file"], mode="r")

        assert isinstance(result, pd.DataFrame)
        assert tuple(result.columns) == tuple(df.columns)
        assert result.shape == df.shape

        assert result[["latitude", "longitude"]].equals(df[["latitude", "longitude"]])
        # limiting case where all z values are in the dataframe
        # Result should be flattened z values corresponding to each (latitude, longitude) pair
        assert np.allclose(result["stalt"], [1.0, 2.0, 3.0, 4.0])


def test_fill_heights_missing_station_altitude():
    config = {"orography_file": "/path/to/orography.zarr", "station_altitude": "stalt"}
    df = pd.DataFrame(
        {
            "latitude": [-45.0, -45.0, 45.0, 45.0],
            "longitude": [0.0, 180.0, 0.0, 180.0],
            # stalt is missing
        }
    )

    with mock.patch("anemoi.transform.filters.tabular.fill_heights.zarr.open") as open_zarr:
        zarr_dataset = mock_zarr_dataset(
            {
                "lat": [-45.0, 45.0],
                "lon": [0.0, 180.0],
                "orog": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        open_zarr.return_value = zarr_dataset

        fill_heights = create_filter("fill_orography", **config)
        with pytest.raises(ValueError):
            _ = fill_heights(df.copy())
