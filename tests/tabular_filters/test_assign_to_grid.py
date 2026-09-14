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

from anemoi.transform.filters import create_filter_by_name as create_filter


def test_assign_to_grid_healpix():
    config = {
        "grid": "h16",
    }
    df = pd.DataFrame(
        {
            "latitude": [-89.9, -89.9, -89.9, 0.0, 0.0, 0.0, 89.9, 89.9, 89.9],
            "longitude": [0.1, 180.0, 359.9, 0.1, 180.0, 359.9, 0.1, 180.0, 359.9],
        }
    )
    assign_to_grid = create_filter("assign_to_grid", **config)
    result = assign_to_grid(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("grid_index_h16", "distance")
    assert result.shape == (len(df), len(df.columns) + 2)
    expected = {
        "grid_index_h16": [3032, 3040, 3047, 1440, 1472, 1567, 24, 32, 39],
        "distance": [
            16.101259,
            16.170669,
            16.101259,
            2.390108,
            2.388015,
            2.7125,
            16.101259,
            16.170669,
            16.101259,
        ],
    }
    for col_name, expected_values in expected.items():
        assert np.allclose(result[col_name].to_numpy(), expected_values)


def test_assign_to_grid_o96():
    config = {
        "grid": "o96",
    }
    df = pd.DataFrame(
        {
            "latitude": [-89.9, -89.9, -89.9, 0.0, 0.0, 0.0, 89.9, 89.9, 89.9],
            "longitude": [0.1, 180.0, 359.9, 0.1, 180.0, 359.9, 0.1, 180.0, 359.9],
        }
    )
    assign_to_grid = create_filter("assign_to_grid", **config)
    result = assign_to_grid(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("grid_index_o96", "distance")
    assert result.shape == (len(df), len(df.columns) + 2)
    expected = {
        "grid_index_o96": [40300, 40310, 40139, 20160, 20360, 20559, 0, 10, 223],
        "distance": [
            0.623840,
            0.615772,
            10.194240,
            0.478106,
            0.467531,
            0.926599,
            0.623840,
            0.615772,
            10.194240,
        ],
    }
    for col_name, expected_values in expected.items():
        assert np.allclose(result[col_name].to_numpy(), expected_values)


def test_assign_to_grid_no_grid():
    config = {"grid": ""}
    with pytest.raises(ValueError):
        _ = create_filter("assign_to_grid", **config)
