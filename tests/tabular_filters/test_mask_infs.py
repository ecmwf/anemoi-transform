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


def test_mask_infs_with_prefix():
    config = {"column_prefix": "col"}
    df = pd.DataFrame({"col1": [np.inf, 1, 2, -np.inf], "col2": [3, np.inf, -np.inf, 6]})
    mask_infs = create_filter("mask_infs", **config)
    result = mask_infs(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    expected_result = pd.DataFrame({"col1": [np.nan, 1, 2, np.nan], "col2": [3, np.nan, np.nan, 6]})
    assert result.equals(expected_result)


def test_mask_infs_with_prefix_missing():
    config = {"column_prefix": "col"}
    df = pd.DataFrame({"foo": [np.inf, 1, 2, -np.inf], "bar": [3, np.inf, -np.inf, 6]})
    mask_infs = create_filter("mask_infs", **config)
    with pytest.raises(ValueError):
        _ = mask_infs(df.copy())


def test_mask_infs_with_columns():
    config = {"columns": ["col1"]}
    df = pd.DataFrame({"col1": [np.inf, 1, 2, -np.inf], "col2": [3, np.inf, -np.inf, 6]})
    mask_infs = create_filter("mask_infs", **config)
    result = mask_infs(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([np.nan, 1.0, 2.0, np.nan], name="col1")
    assert result["col1"].equals(expected_result)


def test_mask_infs_with_columns_missing():
    config = {"columns": ["col1"]}
    df = pd.DataFrame(
        {
            # col1 missing
            "col2": [3, np.inf, -np.inf, 6]
        }
    )
    mask_infs = create_filter("mask_infs", **config)
    with pytest.raises(ValueError):
        _ = mask_infs(df.copy())


def test_mask_infs_with_both_column_specs():
    config = {"columns": ["col1"], "column_prefix": "col"}
    with pytest.raises(ValueError):
        _ = create_filter("mask_infs", **config)


def test_mask_infs_with_no_column_specs():
    config = {}
    with pytest.raises(ValueError):
        _ = create_filter("mask_infs", **config)
