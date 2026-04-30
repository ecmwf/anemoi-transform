# (C) Copyright 2026- Anemoi contributors.
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


def test_impute_nans_scalar_all_columns():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan],
            "z": [0.0, 1.0, 2.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans_tabular", value=0.0)
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert not result.isnull().any().any()
    assert result.loc[1, "obsvalue_x"] == 0.0
    assert result.loc[2, "obsvalue_y"] == 0.0
    assert result.loc[3, "z"] == 0.0


def test_impute_nans_dict_value():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0],
            "obsvalue_y": [0.0, 1.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans_tabular", value={"obsvalue_x": -1.0, "obsvalue_y": -2.0})
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.loc[1, "obsvalue_x"] == -1.0
    assert result.loc[2, "obsvalue_y"] == -2.0
    assert not result.isnull().any().any()


def test_impute_nans_with_columns_validates_presence():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0],
            "z": [0.0, 1.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans_tabular", value=0.0, columns=["obsvalue_x"])
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape


def test_impute_nans_with_columns_raises_for_missing():
    df = pd.DataFrame({"z": [1.0, 2.0, np.nan]})
    impute_nans = create_filter("impute_nans_tabular", value=0.0, columns=["obsvalue_x"])
    with pytest.raises(ValueError):
        impute_nans(df.copy())


def test_impute_nans_with_column_prefix():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0],
            "obsvalue_y": [np.nan, 1.0, np.nan],
            "z": [0.0, 1.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans_tabular", value=99.0, column_prefix="obsvalue_")
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert not result[["obsvalue_x", "obsvalue_y"]].isnull().any().any()


def test_impute_nans_with_column_prefix_raises_for_missing():
    df = pd.DataFrame({"z": [1.0, 2.0, np.nan]})
    impute_nans = create_filter("impute_nans_tabular", value=0.0, column_prefix="obsvalue_")
    with pytest.raises(ValueError):
        impute_nans(df.copy())


def test_impute_nans_both_column_specs_raises():
    with pytest.raises(ValueError):
        create_filter("impute_nans_tabular", value=0.0, columns=["obsvalue_x"], column_prefix="obsvalue_")
