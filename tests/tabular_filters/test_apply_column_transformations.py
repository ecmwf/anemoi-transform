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


def test_apply_column_transformations():
    config = {
        "col1": {"function": "log"},
        "col2": {"function": "log1p"},
        "col3": {"function": "sqrt"},
        "col4": {"function": "exp"},
        "col5": {"function": "abs"},
        "col6": {"function": "sin"},
        "col7": {"function": "cos"},
        "col8": {"function": "lambda x: x + 1"},
    }
    df = pd.DataFrame(
        {
            "col1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col2": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col3": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col4": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col5": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col6": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col7": [0.0, 1.0, 2.0, 3.0, 4.0],
            "col8": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    apply_column_transformations = create_filter("apply_column_transformations", **config)
    result = apply_column_transformations(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    for col_name, spec in config.items():
        func_str = spec["function"]
        if "lambda" in func_str:
            oper = eval(func_str)
        else:
            oper = getattr(np, func_str)
        expected = oper(df[col_name].to_numpy())
        assert np.allclose(result[col_name].to_numpy(), expected, equal_nan=True)


def test_add_sine():
    # test that apply_column_transformations can replace legacy "add_sine" filter
    config = {
        "sin_col1": {"function": "sin_deg", "source_column": "col1"},
    }
    df = pd.DataFrame(
        {
            "col1": [0.0, 90.0, 180.0, 270.0, 360.0],
            "col2": [0, 1, 2, 3, 4],
        }
    )
    add_sine = create_filter("apply_column_transformations", **config)
    result = add_sine(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("sin_col1",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["col1", "col2"]].equals(df[["col1", "col2"]])

    expected_values = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    assert np.allclose(result["sin_col1"].to_numpy(), expected_values)


def test_add_cosine():
    # test that apply_column_transformations can replace legacy "add_cosine" filter
    config = {
        "cos_col1": {"function": "cos_deg", "source_column": "col1"},
    }
    df = pd.DataFrame(
        {
            "col1": [0.0, 90.0, 180.0, 270.0, 360.0],
            "col2": [0, 1, 2, 3, 4],
        }
    )
    add_cosine = create_filter("apply_column_transformations", **config)
    result = add_cosine(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("cos_col1",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["col1", "col2"]].equals(df[["col1", "col2"]])

    expected_values = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
    assert np.allclose(result["cos_col1"].to_numpy(), expected_values)


def test_apply_column_transformations_one_source_column_new_column():
    config = {
        "col2": {
            "function": "lambda x: x + 1",
            "source_column": "col1",
        },
    }
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    apply_column_transformations = create_filter("apply_column_transformations", **config)
    result = apply_column_transformations(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("col2",)
    assert result.shape == (df.shape[0], df.shape[1] + 1)

    expected = df["col1"].to_numpy() + 1
    assert np.allclose(result["col2"].to_numpy(), expected, equal_nan=True)
    assert result["col1"].equals(df["col1"])


def test_apply_column_transformations_two_source_columns():
    config = {
        "col2": {
            "function": "lambda x, y: (x + 1)*y",
            "source_column": ["col2", "col1"],
        },
    }
    df = pd.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    apply_column_transformations = create_filter("apply_column_transformations", **config)
    result = apply_column_transformations(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    expected = (df["col2"].to_numpy() + 1) * df["col1"].to_numpy()
    assert np.allclose(result["col2"].to_numpy(), expected, equal_nan=True)
    assert result["col1"].equals(df["col1"])


def test_apply_column_transformations_safe_log():
    config = {
        "log_col1": {
            "function": "safe_log",
            "source_column": "col1",
        }
    }
    df = pd.DataFrame(
        {
            "col1": [0, 1, 2, 3, 4],
        }
    )
    apply_column_transformations = create_filter("apply_column_transformations", **config)
    result = apply_column_transformations(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(list(df.columns) + ["log_col1"])
    assert result.shape == (len(df), len(df.columns) + 1)

    expected = np.log(df["col1"].to_numpy() + 1e-10)
    assert np.allclose(result["log_col1"].to_numpy(), expected)


def test_apply_column_transformations_missing_column():
    config = {
        "col1": {"function": "log"},
        "col2": {"function": "log1p"},
    }
    df = pd.DataFrame(
        {
            "col1": [0.0, 1.0, 2.0, 3.0, 4.0],
            # col2 is missing
        }
    )
    apply_column_transformations = create_filter("apply_column_transformations", **config)
    with pytest.raises(KeyError):
        _ = apply_column_transformations(df.copy())


def test_apply_column_transformations_bad_transformation():
    config = {
        "col": {"function": "unknown_function"},
    }
    with pytest.raises(ValueError):
        _ = create_filter("apply_column_transformations", **config)


def test_apply_column_transformations_no_config():
    with pytest.raises(ValueError):
        _ = create_filter("apply_column_transformations")
