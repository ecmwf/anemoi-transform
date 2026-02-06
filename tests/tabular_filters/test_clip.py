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


def test_clip():
    config = {
        "col1": (1, 2),
    }
    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    clip = create_filter("clip", **config)
    result = clip(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([1, 1, 2, 2], name="col1")
    assert result["col1"].equals(expected_result)


def test_clip_no_upper():
    config = {
        "col1": (1, None),
    }
    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    clip = create_filter("clip", **config)
    result = clip(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([1, 1, 2, 3], name="col1")
    assert result["col1"].equals(expected_result)


def test_clip_no_lower():
    config = {
        "col1": (None, 2),
    }
    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    clip = create_filter("clip", **config)
    result = clip(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([0, 1, 2, 2], name="col1")
    assert result["col1"].equals(expected_result)


def test_clip_missing_col():
    config = {
        "col1": (1, 2),
    }
    # df missing col1
    df = pd.DataFrame({"col2": [3, 4, 5, 6]})
    clip = create_filter("clip", **config)
    with pytest.raises(ValueError):
        _ = clip(df.copy())


def test_clip_no_config():
    with pytest.raises(ValueError):
        _ = create_filter("clip")
