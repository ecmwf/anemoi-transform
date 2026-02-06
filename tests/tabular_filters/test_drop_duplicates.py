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


def test_drop_duplicates():
    config = {
        "columns": ["y", "z"],
    }
    df = pd.DataFrame(
        {
            "x": [0, 1, 1, 1, 0, 0],
            "y": [0, 1, 1, 0, 0, 1],
            "z": [0, 0, 1, 1, 0, 1],
        }
    )
    drop_duplicates = create_filter("drop_duplicates", **config)
    result = drop_duplicates(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop rows such that the entire row is unique (across selected columns)
    # (in this case just the last two rows)
    assert result.shape == (4, len(df.columns))

    assert result.equals(df.drop(index=[4, 5]))


def test_drop_duplicates_unknown_column():
    config = {
        "columns": ["x"],
    }
    df = pd.DataFrame(
        {
            # x is missing
            "y": [0, 1, 1, 0, 0, 1],
            "z": [0, 0, 1, 1, 0, 1],
        }
    )
    drop_duplicates = create_filter("drop_duplicates", **config)
    with pytest.raises(ValueError):
        _ = drop_duplicates(df.copy())


def test_drop_duplicates_with_prefix():
    config = {
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "x": [0, 1, 1, 1, 0, 0],
            "obsvalue_y": [0, 1, 1, 0, 0, 1],
            "obsvalue_z": [0, 0, 1, 1, 0, 1],
        }
    )
    drop_duplicates = create_filter("drop_duplicates", **config)
    result = drop_duplicates(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop rows such that the entire row is unique (across selected columns)
    # (in this case just the last two rows)
    assert result.shape == (4, len(df.columns))

    assert result.equals(df.drop(index=[4, 5]))


def test_drop_duplicates_no_config():
    df = pd.DataFrame(
        {
            "x": [0, 1, 1, 1, 0, 0],
            "y": [0, 1, 1, 0, 0, 1],
            "z": [0, 0, 1, 1, 0, 1],
        }
    )
    drop_duplicates = create_filter("drop_duplicates")
    result = drop_duplicates(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop rows such that the entire row is unique - no subsetting
    assert result.shape == (5, len(df.columns))

    assert result.equals(df.drop(index=[4]))
