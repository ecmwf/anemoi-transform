# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.transform.filters import filter_registry

from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

U_VALUES = np.array([[-3.26786804, -2.90458679], [-4.28153992, -10.75224304], [-6.29130554, -4.17704773]])
V_VALUES = np.array([[6.51824951, 4.7321167], [1.16961670, 1.73797607], [-2.93096924, 3.2399292]])

ROTATED_U_VALUES = np.array([[-3.22801207, -2.87233492], [-4.28153992, -10.75224304], [-6.27393622, -4.15286741]])
ROTATED_V_VALUES = np.array([[6.53807895, 4.7517623], [1.1696167, 1.73797607], [-2.96796738, 3.27086552]])


@pytest.fixture
def wind_source(test_source):
    WIND_SPEC = [
        {"param": "10u", "values": U_VALUES, **MOCK_FIELD_METADATA},
        {"param": "10v", "values": V_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(WIND_SPEC)


@pytest.fixture
def rotated_wind_source(test_source):
    ROTATED_WIND_SPEC = [
        {"param": "10u", "values": ROTATED_U_VALUES, **MOCK_FIELD_METADATA},
        {"param": "10v", "values": ROTATED_V_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(ROTATED_WIND_SPEC)


def test_rotate_winds(wind_source, rotated_wind_source):
    rotate_winds = filter_registry.create("rotate_winds", x_wind="10u", y_wind="10v")

    pipeline = wind_source | rotate_winds

    input_fields = collect_fields_by_param(wind_source)
    output_fields = collect_fields_by_param(pipeline)

    # Check for expected params
    assert set(input_fields) == {"10u", "10v"}
    assert set(output_fields) == {"10u", "10v"}

    expected_fields = collect_fields_by_param(rotated_wind_source)
    assert set(expected_fields) == set(output_fields)

    for param in ("10u", "10v"):
        assert len(output_fields[param]) == len(expected_fields[param])
        for output_field, expected_field in zip(output_fields[param], expected_fields[param]):
            assert np.allclose(output_field.to_numpy(flatten=True), expected_field.to_numpy(flatten=True))


@pytest.mark.xfail(reason="AttributeError: 'RegularDistinctLLGeography' object has no attribute 'latitudes_unrotated'")
def test_unrotate_winds(rotated_wind_source, wind_source):
    unrotate_winds = filter_registry.create("unrotate_winds", x_wind="10u", y_wind="10v")

    pipeline = rotated_wind_source | unrotate_winds

    input_fields = collect_fields_by_param(rotated_wind_source)
    output_fields = collect_fields_by_param(pipeline)

    # Check for expected params
    assert set(input_fields) == {"10u", "10v"}
    assert set(output_fields) == {"10u", "10v"}

    expected_fields = collect_fields_by_param(wind_source)
    assert set(expected_fields) == set(output_fields)

    for param in ("10u", "10v"):
        assert len(output_fields[param]) == len(expected_fields[param])
        for output_field, expected_field in zip(output_fields[param], expected_fields[param]):
            assert np.allclose(output_field.to_numpy(flatten=True), expected_field.to_numpy(flatten=True))


@pytest.mark.xfail(reason="AttributeError: 'RegularDistinctLLGeography' object has no attribute 'latitudes_unrotated'")
def test_rotate_winds_roundtrip(wind_source):
    rotate_winds = filter_registry.create("rotate_winds", x_wind="10u", y_wind="10v")
    unrotate_winds = filter_registry.create("unrotate_winds", x_wind="10u", y_wind="10v")

    rotated_wind_source = wind_source | rotate_winds
    pipeline = rotated_wind_source | unrotate_winds

    input_fields = collect_fields_by_param(wind_source)
    output_fields = collect_fields_by_param(pipeline)

    # Check for expected params
    assert set(input_fields) == {"10u", "10v"}
    assert set(output_fields) == {"10u", "10v"}

    expected_fields = collect_fields_by_param(rotated_wind_source)
    assert set(expected_fields) == set(output_fields)

    # test input and output are the same
    for param in ("10u", "10v"):
        assert len(input_fields[param]) == len(output_fields[param])
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert np.allclose(input_field.to_numpy(flatten=True), output_field.to_numpy(flatten=True))


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
