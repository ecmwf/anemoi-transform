# (C) Copyright 2025-2026 Anemoi contributors.
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
}

SENTINEL = -999.0

# Two fields of the same parameter whose sentinels sit in different places, plus an
# unrelated parameter that must pass through untouched.
FIELD_SPECS = [
    {
        "param": "cin",
        "valid_datetime": "2018-08-01T09:00:00Z",
        "values": np.array([[SENTINEL, SENTINEL], [12.0, 30.0], [SENTINEL, 4.0]]),
        **MOCK_FIELD_METADATA,
    },
    {
        "param": "cin",
        "valid_datetime": "2018-08-01T10:00:00Z",
        "values": np.array([[SENTINEL, 45.0], [SENTINEL, 30.0], [1.0, 4.0]]),
        **MOCK_FIELD_METADATA,
    },
    {
        "param": "t",
        "valid_datetime": "2018-08-01T09:00:00Z",
        "values": np.array([[SENTINEL, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        **MOCK_FIELD_METADATA,
    },
]


@pytest.fixture()
def source(test_source):
    return test_source([{**spec, "values": spec["values"].copy()} for spec in FIELD_SPECS])


def test_self_mask_is_computed_per_field(source):
    """Each field is masked by its own values, not by the first field's."""
    apply_mask = create_filter("apply_mask", self_mask=True, param="cin", threshold=SENTINEL, threshold_operator="<=")

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(source | apply_mask)

    for input_field, output_field in zip(input_fields["cin"], output_fields["cin"]):
        values = input_field.to_numpy(flatten=True)
        expected = values.copy()
        expected[values <= SENTINEL] = np.nan
        result = output_field.to_numpy(flatten=True)
        assert np.array_equal(expected, result, equal_nan=True)
        assert not np.any(result[~np.isnan(result)] <= SENTINEL)

    # an unselected parameter keeps its sentinel
    assert np.array_equal(input_fields["t"][0].to_numpy(flatten=True), output_fields["t"][0].to_numpy(flatten=True))


def test_self_mask_with_mask_value(source):
    apply_mask = create_filter("mask", self_mask=True, param="cin", mask_value=SENTINEL)

    output_fields = collect_fields_by_param(source | apply_mask)
    for field in output_fields["cin"]:
        result = field.to_numpy(flatten=True)
        assert not np.any(result[~np.isnan(result)] == SENTINEL)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"self_mask": True, "param": "cin", "mask_param": "lsm", "threshold": 0}, "Exactly one of"),
        ({"threshold": 0}, "Exactly one of"),
        ({"self_mask": True, "threshold": 0}, "`param` must be provided"),
        ({"self_mask": True, "param": "cin", "threshold": 0, "return_mask": True}, "`return_mask`"),
        ({"self_mask": True, "param": "cin"}, "Exactly one of `mask_value` or `threshold`"),
    ],
)
def test_self_mask_rejects_invalid_arguments(kwargs, match):

    with pytest.raises(ValueError, match=match):
        create_filter("apply_mask_fields", **kwargs)
