# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime

import numpy as np

from ..utils import collect_fields_by_param
from ..utils import create_fields_filter as create_filter

MOCK_FIELD_METADATA = {
    "geography.distinct_latitudes": [10.0, 0.0, -10.0],
    "geography.distinct_longitudes": [20, 40.0],
}


def _to_datetime(dt_str):
    return datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def test_accum_to_interval_zero_left_true(test_source):
    """Accumulated-from-start fields are differenced into intervals with zero at first step.

    - Works per variable along valid_datetime ordering even if inputs are not sorted.
    - Non-target variables pass through unchanged.
    """
    # Build 3x2 arrays; base increment per step
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Accumulated values at times (sorted times would be 00Z -> 06Z -> 12Z)
    ACC_00 = np.zeros_like(B)
    ACC_06 = B
    ACC_12 = 2 * B

    # Provide inputs out of chronological order to ensure sorting by valid_datetime works
    FIELD_SPECS = [
        {
            "parameter.variable": "tp",
            "data.values": ACC_12,
            "time.valid_datetime": _to_datetime("2018-08-01T12:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        {
            "parameter.variable": "tp",
            "data.values": ACC_00,
            "time.valid_datetime": _to_datetime("2018-08-01T00:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        {
            "parameter.variable": "tp",
            "data.values": ACC_06,
            "time.valid_datetime": _to_datetime("2018-08-01T06:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        # Non-target variable should pass through unchanged
        {
            "parameter.variable": "t",
            "data.values": B + 10,
            "time.valid_datetime": _to_datetime("2018-08-01T00:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        {
            "parameter.variable": "t",
            "data.values": B + 20,
            "time.valid_datetime": _to_datetime("2018-08-01T06:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
    ]

    source = test_source(FIELD_SPECS)
    accum_to_interval = create_filter("accum_to_interval", variables=["tp"], zero_left=True)
    pipeline = source | accum_to_interval

    output_fields = collect_fields_by_param(pipeline)

    # Expect intervals in chronological order: 00Z -> 06Z -> 12Z
    assert set(output_fields) == {"tp", "t"}
    tp_fields = sorted(output_fields["tp"], key=lambda f: f.time.valid_datetime())

    expected_tp = [
        np.zeros_like(B),  # first step zeroed
        ACC_06 - ACC_00,
        ACC_12 - ACC_06,
    ]
    for f, exp in zip(tp_fields, expected_tp):
        assert np.allclose(f.to_numpy(), exp)

    t_inputs = {
        spec["time.valid_datetime"]: spec["data.values"] for spec in FIELD_SPECS if spec["parameter.variable"] == "t"
    }
    for f in output_fields["t"]:
        ts = f.time.valid_datetime()
        assert np.allclose(f.to_numpy(), t_inputs[ts])


def test_accum_to_interval_zero_left_false(test_source):
    """First step retains its accumulated value when zero_left is False; subsequent steps are differences."""
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Accumulated values (monotonic increasing), sorted times would be 00Z -> 06Z -> 12Z
    ACC_00 = B
    ACC_06 = 2 * B
    ACC_12 = 3 * B

    FIELD_SPECS = [
        {
            "parameter.variable": "tp",
            "data.values": ACC_12,
            "time.valid_datetime": _to_datetime("2018-08-01T12:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        {
            "parameter.variable": "tp",
            "data.values": ACC_06,
            "time.valid_datetime": _to_datetime("2018-08-01T06:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
        {
            "parameter.variable": "tp",
            "data.values": ACC_00,
            "time.valid_datetime": _to_datetime("2018-08-01T00:00:00Z"),
            **MOCK_FIELD_METADATA,
        },
    ]

    source = test_source(FIELD_SPECS)
    accum_to_interval = create_filter("accum_to_interval", variables=["tp"], zero_left=False)
    pipeline = source | accum_to_interval

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"tp"}
    tp_fields = sorted(output_fields["tp"], key=lambda f: f.time.valid_datetime())

    expected_tp = [
        ACC_00,  # first step unchanged when zero_left is False
        ACC_06 - ACC_00,  # interval differences thereafter
        ACC_12 - ACC_06,
    ]
    for f, exp in zip(tp_fields, expected_tp):
        assert np.allclose(f.to_numpy(), exp)
