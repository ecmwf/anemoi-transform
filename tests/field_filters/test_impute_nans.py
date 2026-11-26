# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
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
R_VALUES = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def source(test_source):
    return test_source(
        [
            {"parameter.variable": "t", "data.values": T_VALUES.copy(), **INPUT_METADATA},
            {"parameter.variable": "q", "data.values": Q_VALUES.copy(), **INPUT_METADATA},
            {"parameter.variable": "r", "data.values": R_VALUES.copy(), **INPUT_METADATA},
        ]
    )


def test_impute_nans_single_param(source):
    impute_nans = create_filter("impute_nans", param="t", value=0.0)
    pipeline = source | impute_nans

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"t", "q", "r"}

    # t: NaNs replaced with 0.0
    result = output_fields["t"][0].to_numpy(flatten=True)
    expected = T_VALUES.flatten().copy()
    expected[np.isnan(expected)] = 0.0
    assert np.array_equal(result, expected)
    assert not np.any(np.isnan(result))

    # q and r pass through unchanged
    for param in ("q", "r"):
        result = output_fields[param][0].to_numpy(flatten=True)
        original = input_fields[param][0].to_numpy(flatten=True)
        assert np.array_equal(result, original, equal_nan=True)


def test_impute_nans_multiple_params(source):
    impute_nans = create_filter("impute_nans", param=["t", "q"], value=-1.0)
    pipeline = source | impute_nans

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # t and q: NaNs replaced
    for param, values in (("t", T_VALUES), ("q", Q_VALUES)):
        result = output_fields[param][0].to_numpy(flatten=True)
        expected = values.flatten().copy()
        expected[np.isnan(expected)] = -1.0
        assert np.array_equal(result, expected)
        assert not np.any(np.isnan(result))

    # r passes through unchanged
    result = output_fields["r"][0].to_numpy(flatten=True)
    original = input_fields["r"][0].to_numpy(flatten=True)
    assert np.array_equal(result, original, equal_nan=True)


def test_impute_nans_no_nans_unchanged(source):
    impute_nans = create_filter("impute_nans", param="r", value=0.0)
    pipeline = source | impute_nans

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # r has no NaNs — output should be identical to input
    result = output_fields["r"][0].to_numpy(flatten=True)
    original = input_fields["r"][0].to_numpy(flatten=True)
    assert np.array_equal(result, original)


def test_impute_nans_grid_unchanged(source):
    impute_nans = create_filter("impute_nans", param="t", value=0.0)
    pipeline = source | impute_nans

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # grid should be unchanged (unlike remove_nans which reduces the grid)
    input_lats, input_lons = input_fields["t"][0].geography.latlons()
    output_lats, output_lons = output_fields["t"][0].geography.latlons()
    assert np.array_equal(input_lats, output_lats)
    assert np.array_equal(input_lons, output_lons)


def test_impute_nans_replace_nans_alias(source):
    impute_nans = create_filter("replace_nans", param="t", value=0.0)
    pipeline = source | impute_nans

    output_fields = collect_fields_by_param(pipeline)

    result = output_fields["t"][0].to_numpy(flatten=True)
    assert not np.any(np.isnan(result))


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
