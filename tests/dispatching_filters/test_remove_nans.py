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

from anemoi.transform.filters import dispatching_filter_registry as filter_registry

from ..utils import collect_fields_by_param

INPUT_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 30.0, 40.0],
    "valid_datetime": "2018-08-01T12:00:00Z",
}

INPUT_VALUES = [
    np.array([[1.0, np.nan, 20.0], [np.nan, 3.0, np.nan], [4.0, 4.5, np.nan]]),
    # fewer NaNs than the first field
    np.array([[1.0, 1.5, 21.0], [np.nan, 3.0, np.nan], [4.0, 4.5, 5.0]]),
    # more NaNs than the first field
    np.array([[np.nan, np.nan, 22.0], [np.nan, 3.0, np.nan], [4.0, 4.5, np.nan]]),
]

EXPECTED_VALUES = [
    # mask generated from NaNs in the first field is applied
    np.array([1.0, 20.0, 3.0, 4.0, 4.5]),
    np.array([1.0, 21.0, 3.0, 4.0, 4.5]),
    np.array([np.nan, 22.0, 3.0, 4.0, 4.5]),
]

EXPECTED_METADATA = {
    # take the original (flattened) versions and remove where there were NaNs in the first field
    # "latitudes": [10.0, ---, 10.0, ---, 0.0, ---, -10.0, -10.0, ---],
    "latitudes": [10.0, 10.0, 0.0, -10.0, -10.0],
    # "longitudes": [20.0, ---, 40.0, ---, 30.0, ---, 20.0, 30.0, ---],
    "longitudes": [20.0, 40.0, 30.0, 20.0, 30.0],
}


@pytest.fixture
def source(test_source):
    FIELD_SPECS = [
        {"param": "t", "step": i, "values": values.copy(), **INPUT_METADATA} for i, values in enumerate(INPUT_VALUES)
    ]
    return test_source(FIELD_SPECS)


def test_remove_nans_fields(source):
    remove_nans = filter_registry.create("remove_nans")
    pipeline = source | remove_nans

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    param = "t"
    assert set(input_fields) == {param}
    assert set(input_fields) == set(output_fields)
    assert len(input_fields[param]) == len(output_fields[param])

    for i, (input_field, output_field) in enumerate(zip(input_fields[param], output_fields[param])):
        # existing behaviour is to generate a mask based on the location of
        # NaNs in the first field and apply this mask for all fields
        assert np.array_equal(input_field.to_numpy(flatten=True), INPUT_VALUES[i].flatten(), equal_nan=True)
        assert np.array_equal(output_field.to_numpy(flatten=True), EXPECTED_VALUES[i], equal_nan=True)

        output_lats, output_lons = output_field.grid_points()
        assert np.array_equal(output_lats, EXPECTED_METADATA["latitudes"], equal_nan=True)
        assert np.array_equal(output_lons, EXPECTED_METADATA["longitudes"], equal_nan=True)


def test_drop_nans_tabular():
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
    drop_nans = filter_registry.create("drop_nans", **config)
    result = drop_nans(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    # only drop one row (where both 'obsvalue_' columns are NaN)
    assert result.shape == (len(df) - 1, len(df.columns))

    assert result.equals(df.drop(index=3))
