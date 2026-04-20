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

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

LSM_VALUES = np.array([[1, 0], [1, 1], [0, 0]])

DATA_VALUES = {
    "sd": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "lsm": LSM_VALUES.astype(float),
    "2t": np.array([[7.0, 8.0], [9.0, 0.0], [9.0, 8.0]]),
}


@pytest.fixture()
def source(test_source):
    FIELD_SPECS = [
        {"param": param, "values": values.copy(), **MOCK_FIELD_METADATA} for param, values in DATA_VALUES.items()
    ]
    return test_source(FIELD_SPECS)


def test_apply_mask_from_field_mask_value(source):
    """Test masking using a field from the pipeline with mask_value."""
    apply_mask = create_filter("apply_mask", mask_param="lsm", mask_value=0)

    pipeline = source | apply_mask
    output_fields = collect_fields_by_param(pipeline)

    # lsm should be removed from output
    assert "lsm" not in output_fields

    # sd and 2t should be present and masked
    expected_mask = LSM_VALUES.flatten() == 0
    for param in ("sd", "2t"):
        assert param in output_fields
        for field in output_fields[param]:
            values = field.to_numpy(flatten=True)
            expected = DATA_VALUES[param].flatten().copy()
            expected[expected_mask] = np.nan
            assert np.array_equal(values, expected, equal_nan=True)


def test_apply_mask_from_field_threshold(source):
    """Test masking using a field from the pipeline with threshold."""
    apply_mask = create_filter("apply_mask", mask_param="lsm", threshold=0.5, threshold_operator="<")

    pipeline = source | apply_mask
    output_fields = collect_fields_by_param(pipeline)

    # lsm should be removed from output
    assert "lsm" not in output_fields

    expected_mask = LSM_VALUES.flatten() < 0.5
    for param in ("sd", "2t"):
        assert param in output_fields
        for field in output_fields[param]:
            values = field.to_numpy(flatten=True)
            expected = DATA_VALUES[param].flatten().copy()
            expected[expected_mask] = np.nan
            assert np.array_equal(values, expected, equal_nan=True)


def test_apply_mask_from_field_single_param(source):
    """Test masking only a single param using mask_param."""
    apply_mask = create_filter("apply_mask", mask_param="lsm", mask_value=0, param="sd")

    pipeline = source | apply_mask
    output_fields = collect_fields_by_param(pipeline)

    # lsm should be removed
    assert "lsm" not in output_fields

    # sd should be masked
    expected_mask = LSM_VALUES.flatten() == 0
    for field in output_fields["sd"]:
        values = field.to_numpy(flatten=True)
        expected = DATA_VALUES["sd"].flatten().copy()
        expected[expected_mask] = np.nan
        assert np.array_equal(values, expected, equal_nan=True)

    # 2t should be unchanged
    for field in output_fields["2t"]:
        values = field.to_numpy(flatten=True)
        expected = DATA_VALUES["2t"].flatten()
        assert np.array_equal(values, expected)


def test_apply_mask_from_field_missing_param(source):
    """Test that an error is raised when mask_param is not found in data."""
    apply_mask = create_filter("apply_mask", mask_param="nonexistent", mask_value=0)

    with pytest.raises(ValueError, match="not found in input data"):
        list(source | apply_mask)


def test_apply_mask_fails_without_path_or_mask_param():
    """Test that an error is raised when neither path nor mask_param is provided."""
    with pytest.raises(ValueError, match="Exactly one of `path` or `mask_param`"):
        create_filter("apply_mask_fields", mask_value=0)


def test_apply_mask_fails_with_both_path_and_mask_param():
    """Test that an error is raised when both path and mask_param are provided."""
    with pytest.raises(ValueError, match="Exactly one of `path` or `mask_param`"):
        create_filter("apply_mask", path="some_file", mask_param="lsm", mask_value=0)
