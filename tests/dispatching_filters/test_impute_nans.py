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
    "geography.distinct_latitudes": [10.0, 0.0, -10.0],
    "geography.distinct_longitudes": [20.0, 30.0, 40.0],
    "time.valid_datetime": "2018-08-01T12:00:00Z",
}

T_VALUES = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0], [7.0, np.nan, 9.0]])
Q_VALUES = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, np.nan]])


@pytest.fixture
def source(test_source):
    return test_source(
        [
            {"parameter.variable": "t", "data.values": T_VALUES.copy(), **INPUT_METADATA},
            {"parameter.variable": "q", "data.values": Q_VALUES.copy(), **INPUT_METADATA},
        ]
    )


def test_impute_nans_dispatches_to_field_filter(source):
    impute_nans = create_filter("impute_nans", param="t", value=0.0)
    pipeline = source | impute_nans

    output_fields = collect_fields_by_param(pipeline)

    result = output_fields["t"][0].to_numpy(flatten=True)
    assert not np.any(np.isnan(result))

    # q is unmodified
    input_fields = collect_fields_by_param(source)
    q_result = output_fields["q"][0].to_numpy(flatten=True)
    q_original = input_fields["q"][0].to_numpy(flatten=True)
    assert np.array_equal(q_result, q_original, equal_nan=True)


def test_impute_nans_dispatches_to_tabular_filter():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0, np.nan, 4.0],
            "obsvalue_y": [0.0, 1.0, np.nan, np.nan, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans", columns=["obsvalue_x", "obsvalue_y"], value=0.0)
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert not result[["obsvalue_x", "obsvalue_y"]].isnull().any().any()


def test_impute_nans_tabular_with_prefix():
    df = pd.DataFrame(
        {
            "obsvalue_x": [0.0, np.nan, 2.0],
            "obsvalue_y": [np.nan, 1.0, np.nan],
            "z": [0.0, 1.0, np.nan],
        }
    )
    impute_nans = create_filter("impute_nans", column_prefix="obsvalue_", value=-1.0)
    result = impute_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    assert not result[["obsvalue_x", "obsvalue_y"]].isnull().any().any()


def test_impute_nans_tabular_raises_for_missing_columns():
    df = pd.DataFrame({"z": [1.0, 2.0, np.nan]})
    impute_nans = create_filter("impute_nans", columns=["obsvalue_x"], value=0.0)
    with pytest.raises(ValueError):
        impute_nans(df.copy())


def test_impute_nans_tabular_raises_for_missing_prefix():
    df = pd.DataFrame({"z": [1.0, 2.0, np.nan]})
    impute_nans = create_filter("impute_nans", column_prefix="obsvalue_", value=0.0)
    with pytest.raises(ValueError):
        impute_nans(df.copy())
