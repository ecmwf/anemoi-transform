# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter


def test_geopotential_to_height_inplace_implicit():
    config = {"geopotential": "z"}
    df = pd.DataFrame(
        {
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    geopotential_to_height = create_filter("geopotential_to_height", **config)
    result = geopotential_to_height(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["z"].equals(df["z"] / 9.80665)


def test_geopotential_to_height_inplace_explicit():
    config = {
        "geopotential": "z",
        "height": "z",
    }
    df = pd.DataFrame(
        {
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    geopotential_to_height = create_filter("geopotential_to_height", **config)
    result = geopotential_to_height(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["z"].equals(df["z"] / 9.80665)


def test_geopotential_to_height_new_col():
    config = {
        "geopotential": "z",
        "height": "height",
    }
    df = pd.DataFrame(
        {
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    geopotential_to_height = create_filter("geopotential_to_height", **config)
    result = geopotential_to_height(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("height",)
    assert result.shape == (df.shape[0], df.shape[1] + 1)

    assert result["height"].equals(df["z"] / 9.80665)


def test_geopotential_to_height_missing_column():
    config = {
        "geopotential": "geopotential",
    }
    df = pd.DataFrame(
        {
            # geopotential column missing - wrong name
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    geopotential_to_height = create_filter("geopotential_to_height", **config)
    with pytest.raises(ValueError):
        _ = geopotential_to_height(df.copy())
