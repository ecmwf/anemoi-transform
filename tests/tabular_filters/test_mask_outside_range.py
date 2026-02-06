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

from tests.utils import create_tabular_filter as create_filter


def test_mask_outside_range():
    config = {
        "col1": [1, 2],
    }

    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    mask_outside_range = create_filter("mask_outside_range", **config)
    result = mask_outside_range(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([np.nan, 1, 2, np.nan], name="col1")
    assert result["col1"].equals(expected_result)


def test_mask_outside_range_no_lower_bound():
    config = {
        "col1": [None, 2],
    }

    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    mask_outside_range = create_filter("mask_outside_range", **config)
    result = mask_outside_range(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([0, 1, 2, np.nan], name="col1")
    assert result["col1"].equals(expected_result)


def test_mask_outside_range_no_upper_bound():
    config = {
        "col1": [1, None],
    }

    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    mask_outside_range = create_filter("mask_outside_range", **config)
    result = mask_outside_range(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([np.nan, 1, 2, 3], name="col1")
    assert result["col1"].equals(expected_result)
