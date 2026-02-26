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


def test_drop_nans_all_with_prefix():
    config = {
        "column_prefix": "obsvalue_",
        "how": "all",
    }
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # only drop one row (where both 'obsvalue_' columns are NaN)
    assert result.shape == (len(df) - 1, len(df.columns))

    assert result.equals(df.drop(index=3))


def test_drop_nans_all_with_columns():
    config = {
        "columns": ["obsvalue_x", "obsvalue_y"],
        "how": "all",
    }
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # only drop one row (where both 'obsvalue_' columns are NaN)
    assert result.shape == (len(df) - 1, len(df.columns))

    assert result.equals(df.drop(index=3))


def test_drop_nans_any_with_columns():
    config = {
        "columns": ["obsvalue_x", "obsvalue_y", "z"],
        "how": "any",
    }
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop any row where any column is NaN
    assert result.shape == (1, len(df.columns))

    assert result.equals(df.drop(index=[1, 2, 3, 4]))


def test_drop_nans_any_with_prefix():
    config = {
        "column_prefix": "obsvalue_",
        "how": "any",
    }
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop any row where any 'obsvalue_' column is NaN
    assert result.shape == (2, len(df.columns))

    assert result.equals(df.drop(index=[1, 2, 3]))


def test_drop_any_nans():
    config = {}
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # drop any row where any column is NaN
    assert result.shape == (1, len(df.columns))

    assert result.equals(df.drop(index=[1, 2, 3, 4]))


def test_drop_nans_with_prefix_missing():
    config = {
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    with pytest.raises(ValueError):
        _ = drop_nans(df.copy())


def test_drop_nans_with_columns_missing():
    config = {
        "columns": ["obsvalue_x", "obsvalue_y"],
    }
    df = pd.DataFrame(
        {
            # obsvalue columns missing
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    drop_nans = create_filter("drop_nans", **config)
    with pytest.raises(ValueError):
        _ = drop_nans(df.copy())


def test_drop_nans_with_both_column_specs():
    config = {
        "column_prefix": "obsvalue_",
        "columns": ["obsvalue_x", "obsvalue_y"],
    }
    with pytest.raises(ValueError):
        _ = create_filter("drop_nans", **config)
