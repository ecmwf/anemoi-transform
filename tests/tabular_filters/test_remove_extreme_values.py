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


def test_remove_extreme_values_drop_with_prefix():
    config = {
        "method": "drop",
        "threshold": 1e5,
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "obsvalue_x": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    # only drop rows where latitude, longitude, obsvalue_* values are *above* threshold
    remove_extreme_values_drop = create_filter("remove_extreme_values", **config)
    result = remove_extreme_values_drop(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == (1, len(df.columns))

    assert result.equals(df.drop(index=[1, 2, 3]))


def test_remove_extreme_values_drop_with_prefix_missing():
    config = {
        "method": "drop",
        "threshold": 1e5,
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    remove_extreme_values_drop = create_filter("remove_extreme_values", **config)
    with pytest.raises(ValueError):
        _ = remove_extreme_values_drop(df.copy())


def test_remove_extreme_values_drop_with_columns():
    config = {
        "method": "drop",
        "threshold": 1e5,
        "columns": ["obsvalue_x"],
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "obsvalue_x": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    # only drop rows where latitude, longitude, obsvalue_x values are *above* threshold
    remove_extreme_values_drop = create_filter("remove_extreme_values", **config)
    result = remove_extreme_values_drop(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == (1, len(df.columns))

    assert result.equals(df.drop(index=[1, 2, 3]))


def test_remove_extreme_values_drop_with_columns_missing():
    config = {
        "method": "drop",
        "threshold": 1e5,
        "columns": ["obsvalue_x"],
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    remove_extreme_values_drop = create_filter("remove_extreme_values", **config)
    with pytest.raises(ValueError):
        _ = remove_extreme_values_drop(df.copy())


def test_remove_extreme_values_drop_with_both_column_specs():
    config = {
        "method": "drop",
        "threshold": 1e5,
        "columns": ["obsvalue_x"],
        "column_prefix": "obsvalue_",
    }
    with pytest.raises(ValueError):
        _ = create_filter("remove_extreme_values", **config)


def test_remove_extreme_values_drop_with_no_column_specs():
    config = {
        "method": "drop",
        "threshold": 1e5,
    }
    with pytest.raises(ValueError):
        _ = create_filter("remove_extreme_values", **config)


def test_remove_extreme_values_mask_with_prefix():
    config = {
        "method": "mask",
        "threshold": 1e5,
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "obsvalue_x": [2.0, 1e6, 1.0, 0.0],
            "fg_deparx": [2.0, 1e6, 1.0, 0.0],
            "biascorr_fgx": [2.0, 1e6, 1.0, 0.0],
            "pressurex": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    # only mask rows where latitude, longitude, obsvalue_* values are *above* threshold
    expected = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, np.nan],
            "longitude": [0.0, 1.0, np.nan, 1.0],
            "obsvalue_x": [2.0, np.nan, 1.0, 0.0],
            "fg_deparx": [2.0, 1e6, 1.0, 0.0],
            "biascorr_fgx": [2.0, 1e6, 1.0, 0.0],
            "pressurex": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    remove_extreme_values_mask = create_filter("remove_extreme_values", **config)
    result = remove_extreme_values_mask(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.equals(expected)


def test_remove_extreme_values_mask_with_prefix_missing():
    config = {
        "method": "mask",
        "threshold": 1e5,
        "column_prefix": "obsvalue_",
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            # obsvalue_x missing
            "fg_deparx": [2.0, 1e6, 1.0, 0.0],
            "biascorr_fgx": [2.0, 1e6, 1.0, 0.0],
            "pressurex": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    remove_extreme_values_mask = create_filter("remove_extreme_values", **config)
    with pytest.raises(ValueError):
        _ = remove_extreme_values_mask(df.copy())


def test_remove_extreme_values_mask_with_columns():
    config = {
        "method": "mask",
        "threshold": 1e5,
        "columns": ["obsvalue_x", "fg_deparx", "biascorr_fgx", "pressurex"],
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            "obsvalue_x": [2.0, 1e6, 1.0, 0.0],
            "fg_deparx": [2.0, 1e6, 1.0, 0.0],
            "biascorr_fgx": [2.0, 1e6, 1.0, 0.0],
            "pressurex": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    expected = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, np.nan],
            "longitude": [0.0, 1.0, np.nan, 1.0],
            "obsvalue_x": [2.0, np.nan, 1.0, 0.0],
            "fg_deparx": [2.0, np.nan, 1.0, 0.0],
            "biascorr_fgx": [2.0, np.nan, 1.0, 0.0],
            "pressurex": [2.0, np.nan, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    remove_extreme_values_mask = create_filter("remove_extreme_values", **config)
    result = remove_extreme_values_mask(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.equals(expected)


def test_remove_extreme_values_mask_with_columns_missing():
    config = {
        "method": "mask",
        "threshold": 1e5,
        "columns": ["obsvalue_x"],
    }
    df = pd.DataFrame(
        {
            "latitude": [0.0, 1e4, 1e5, 1e6],
            "longitude": [0.0, 1.0, 1e6, 1.0],
            # obsvalue_x missing
            # "obsvalue_x": [2.0, 1e6, 1.0, 0.0],
            "fg_deparx": [2.0, 1e6, 1.0, 0.0],
            "biascorr_fgx": [2.0, 1e6, 1.0, 0.0],
            "pressurex": [2.0, 1e6, 1.0, 0.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )

    remove_extreme_values_mask = create_filter("remove_extreme_values", **config)
    with pytest.raises(ValueError):
        _ = remove_extreme_values_mask(df.copy())


def test_remove_extreme_values_mask_with_both_column_specs():
    config = {
        "method": "mask",
        "threshold": 1e5,
        "columns": ["obsvalue_x"],
        "column_prefix": "obsvalue_",
    }
    with pytest.raises(ValueError):
        _ = create_filter("remove_extreme_values", **config)


def test_remove_extreme_values_mask_with_no_column_specs():
    config = {
        "method": "mask",
        "threshold": 1e5,
    }
    with pytest.raises(ValueError):
        _ = create_filter("remove_extreme_values", **config)
