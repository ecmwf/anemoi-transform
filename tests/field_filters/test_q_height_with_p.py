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
from earthkit.meteo import thermo

from anemoi.transform.filters import create_filter_by_name as create_filter

from ..utils import SelectFieldSource
from ..utils import assert_fields_equal
from ..utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "geography.distinct_latitudes": [10.0, 0.0, -10.0],
    "geography.distinct_longitudes": [20.0, 40.0],
    "time.valid_datetime": "2018-08-01T09:00:00Z",
}

T_VALUES = np.array([[280.0, 290.0], [295.0, 285.0], [270.0, 300.0]])
Q_VALUES = np.array([[0.005, 0.010], [0.015, 0.008], [0.002, 0.020]])
P_VALUES = np.array([[95000.0, 100000.0], [101325.0, 92000.0], [88000.0, 98000.0]])

# Derived values — computed from T, Q, P using the same thermo functions as the filters
R_VALUES = thermo.relative_humidity_from_specific_humidity(t=T_VALUES, q=Q_VALUES, p=P_VALUES)


@pytest.fixture
def specific_humidity_source(test_source):
    return test_source(
        [
            {"parameter.variable": "q", "data.values": Q_VALUES.copy(), **MOCK_FIELD_METADATA},
            {"parameter.variable": "t", "data.values": T_VALUES.copy(), **MOCK_FIELD_METADATA},
            {"parameter.variable": "pres", "data.values": P_VALUES.copy(), **MOCK_FIELD_METADATA},
        ]
    )


@pytest.fixture
def relative_humidity_source(test_source):
    return test_source(
        [
            {"parameter.variable": "r", "data.values": R_VALUES.copy(), **MOCK_FIELD_METADATA},
            {"parameter.variable": "t", "data.values": T_VALUES.copy(), **MOCK_FIELD_METADATA},
            {"parameter.variable": "pres", "data.values": P_VALUES.copy(), **MOCK_FIELD_METADATA},
        ]
    )


def test_q_to_r_height_with_p(specific_humidity_source):
    q_to_r = create_filter("q_to_r_height_with_p")
    pipeline = specific_humidity_source | q_to_r

    input_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"q", "t", "pres"}
    assert set(output_fields) == {"q", "t", "pres", "r"}

    # input fields pass through unchanged
    for param in ("q", "t", "pres"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # output matches expected values
    result = output_fields["r"][0].to_numpy()
    np.testing.assert_allclose(result, R_VALUES)


def test_q_to_r_height_with_p_round_trip(specific_humidity_source):
    q_to_r = create_filter("q_to_r_height_with_p")
    r_to_q = create_filter("r_to_q_height_with_p")

    relative_humidity_source = SelectFieldSource(specific_humidity_source | q_to_r, params=["r", "t", "pres"])
    pipeline = relative_humidity_source | r_to_q

    input_fields = collect_fields_by_param(specific_humidity_source)
    intermediate_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"q", "t", "pres"}
    assert set(intermediate_fields) == {"r", "t", "pres"}
    assert set(output_fields) == {"r", "t", "pres", "q"}

    # t and pres unchanged end-to-end
    for param in ("t", "pres"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # intermediate fields unchanged through second filter
    for param in ("r", "t", "pres"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


def test_r_to_q_height_with_p(relative_humidity_source):
    r_to_q = create_filter("r_to_q_height_with_p")
    pipeline = relative_humidity_source | r_to_q

    input_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"r", "t", "pres"}
    assert set(output_fields) == {"r", "t", "pres", "q"}

    # input fields pass through unchanged
    for param in ("r", "t", "pres"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # output matches expected values
    result = output_fields["q"][0].to_numpy()
    expected_q = thermo.specific_humidity_from_relative_humidity(t=T_VALUES, r=R_VALUES, p=P_VALUES)
    np.testing.assert_allclose(result, expected_q)


def test_r_to_q_height_with_p_round_trip(relative_humidity_source):
    r_to_q = create_filter("r_to_q_height_with_p")
    q_to_r = create_filter("q_to_r_height_with_p")

    specific_humidity_source = SelectFieldSource(relative_humidity_source | r_to_q, params=["q", "t", "pres"])
    pipeline = specific_humidity_source | q_to_r

    input_fields = collect_fields_by_param(relative_humidity_source)
    intermediate_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"r", "t", "pres"}
    assert set(intermediate_fields) == {"q", "t", "pres"}
    assert set(output_fields) == {"q", "t", "pres", "r"}

    # t and pres unchanged end-to-end
    for param in ("t", "pres"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # intermediate fields unchanged through second filter
    for param in ("q", "t", "pres"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
