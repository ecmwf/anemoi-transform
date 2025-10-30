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

from .utils import assert_fields_equal
from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

RAD_VALUES = np.array([[2.67687254, 2.59108576], [1.83746659, 1.73104875], [1.1348185, 2.23051268]])

COS_RAD_VALUES = np.array([[-0.89394704, -0.85225947], [-0.26352086, -0.15956740], [0.42229696, -0.61289275]])

SIN_RAD_VALUES = np.array([[0.44817262, 0.52311930], [0.96465370, 0.98718704], [0.90645754, 0.79016611]])


@pytest.fixture
def RAD_source(test_source):
    RAD_SPEC = [
        {"param": "RAD", "values": RAD_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(RAD_SPEC)


@pytest.fixture
def DEG_source(test_source):
    DEG_SPEC = [
        {"param": "DEG", "values": np.rad2deg(RAD_VALUES), **MOCK_FIELD_METADATA},
    ]
    return test_source(DEG_SPEC)


@pytest.fixture
def cos_sin_RAD_source(test_source):
    COS_SIN_RAD = [
        {"param": "cos_RAD", "values": COS_RAD_VALUES, **MOCK_FIELD_METADATA},
        {"param": "sin_RAD", "values": SIN_RAD_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(COS_SIN_RAD)


def test_forward(RAD_source):
    """Test the cos_sin_from_rad filter."""
    filter = filter_registry.create(
        "cos_sin_from_rad",
        param="RAD",
    )
    pipeline = RAD_source | filter

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"cos_RAD", "sin_RAD"}
    assert len(output_fields["cos_RAD"]) == 1
    assert len(output_fields["sin_RAD"]) == 1

    np.testing.assert_allclose(output_fields["cos_RAD"][0].to_numpy(), COS_RAD_VALUES)
    np.testing.assert_allclose(output_fields["sin_RAD"][0].to_numpy(), SIN_RAD_VALUES)


def test_reverse(cos_sin_RAD_source):
    """Test the cos_sin_from_rad filter in reverse."""
    filter = filter_registry.create(
        "cos_sin_from_rad",
        param="some_rad",
        cos_param="cos_RAD",
        sin_param="sin_RAD",
    ).reverse()
    pipeline = cos_sin_RAD_source | filter

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"some_rad"}
    assert len(output_fields["some_rad"]) == 1

    np.testing.assert_allclose(output_fields["some_rad"][0].to_numpy(), RAD_VALUES)


def test_round_trip(RAD_source):
    """Test the cos_sin_from_rad filter reproduces inputs on a round trip."""
    filter = filter_registry.create(
        "cos_sin_from_rad",
        param="RAD",
    )
    cos_sin_RAD_source = RAD_source | filter
    pipeline = cos_sin_RAD_source | filter.reverse()

    input_fields = collect_fields_by_param(RAD_source)
    intermediate_fields = collect_fields_by_param(cos_sin_RAD_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"RAD"}
    assert set(intermediate_fields) == {"cos_RAD", "sin_RAD"}
    assert set(output_fields) == {"RAD"}

    for input_field, output_field in zip(input_fields["RAD"], output_fields["RAD"]):
        assert_fields_equal(input_field, output_field)


def test_exception(DEG_source):
    """Test the cos_sin_from_rad exception.

    Inpupt data in degrees.
    """
    filter = filter_registry.create(
        "cos_sin_from_rad",
        param="DEG",
    )
    pipeline = DEG_source | filter

    with pytest.raises(ValueError):
        collect_fields_by_param(pipeline)


if __name__ == "__main__":
    """
    Run all test functions that start with 'test_'.
    """
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
