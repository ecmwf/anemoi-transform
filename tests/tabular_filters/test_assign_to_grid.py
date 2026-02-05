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
        "grid_index_h16": [3032, 3039, 2871, 1440, 1472, 1472, 24, 31, 199],
        "distance": [
            16.101259,
            16.170669,
            186.739335,
            2.390108,
            2.388015,
            179.915849,
            16.101259,
            16.170669,
            186.739335,
        ],
    }
    for col_name, expected_values in expected.items():
        assert np.allclose(result[col_name].to_numpy(), expected_values)


@pytest.mark.skip(reason="update to use anemoi grid definition")
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
        "grid_index_o96": [40300, 40310, 40310, 20160, 20360, 20360, 0, 10, 10],
        "distance": [
            0.623840,
            0.615772,
            179.901054,
            0.478106,
            0.467531,
            179.900608,
            0.623840,
            0.615772,
            179.901054,
        ],
    }
    for col_name, expected_values in expected.items():
        assert np.allclose(result[col_name].to_numpy(), expected_values)


def test_assign_to_grid_no_grid():
    config = {"grid": ""}
    with pytest.raises(ValueError):
        _ = create_filter("assign_to_grid", **config)
