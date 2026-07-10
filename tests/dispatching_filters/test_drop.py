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

from ..utils import collect_fields_by_param

INPUT_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 30.0, 40.0],
    "valid_datetime": "2018-08-01T12:00:00Z",
}

MOCK_VALUES = np.array([1.0, 2.0, 3.0])


@pytest.fixture
def source(test_source):
    FIELD_SPECS = [
        {"param": "t", "levelist": 500, "values": MOCK_VALUES.copy(), **INPUT_METADATA},
        {
            "param": "t",
            "levelist": 850,
            "values": MOCK_VALUES.copy() * 2,
            **INPUT_METADATA,
        },
        {
            "param": "z",
            "levelist": 500,
            "values": MOCK_VALUES.copy() * 3,
            **INPUT_METADATA,
        },
    ]
    return test_source(FIELD_SPECS)


def test_drop_dispatches_to_tabular():
    config = {"columns": ["drop_me"]}
    df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "drop_me": [3, 4, 5],
        }
    )
    drop = create_filter("drop", **config)
    result = drop(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == ("x",)
    assert result["x"].equals(df["x"])


def test_drop_dispatches_to_fields(source):
    drop = create_filter("drop", param="t")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "t" not in output_fields
    assert "z" in output_fields
    assert len(output_fields["z"]) == 1


def test_drop_fields_alias(source):
    drop = create_filter("drop_fields", param="z")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "z" not in output_fields
    assert "t" in output_fields
    assert len(output_fields["t"]) == 2


def test_drop_columns_alias():
    config = {"columns": ["col_a"]}
    df = pd.DataFrame(
        {
            "col_a": [1, 2, 3],
            "col_b": [4, 5, 6],
        }
    )
    drop = create_filter("drop_columns", **config)
    result = drop(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == ("col_b",)


def test_drop_tabular_no_columns():
    config = {"columns": []}
    with pytest.raises(ValueError):
        _ = create_filter("drop", **config)


def test_drop_tabular_missing_column():
    config = {"columns": ["missing"]}
    df = pd.DataFrame({"x": [0, 1, 2]})
    drop = create_filter("drop", **config)
    with pytest.raises(ValueError):
        _ = drop(df.copy())
