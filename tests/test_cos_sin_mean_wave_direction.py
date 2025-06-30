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

MWD_VALUES = np.array([[153.37349864, 148.45827835], [105.27908047, 99.18178736], [65.02031089, 127.79896253]])
COS_MWD_VALUES = np.array([[-0.89394704, -0.85225947], [-0.26352086, -0.15956740], [0.42229696, -0.61289275]])

SIN_MWD_VALUES = np.array([[0.44817262, 0.52311930], [0.96465370, 0.98718704], [0.90645754, 0.79016611]])


@pytest.fixture
def mwd_source(test_source):
    MWD_SPEC = [
        {"param": "mwd", "values": MWD_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(MWD_SPEC)


@pytest.fixture
def cos_sin_mwd_source(test_source):
    COS_SIN_MWD = [
        {"param": "cos_mwd", "values": COS_MWD_VALUES, **MOCK_FIELD_METADATA},
        {"param": "sin_mwd", "values": SIN_MWD_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(COS_SIN_MWD)


def test_cos_sin_mean_wave_direction(mwd_source):
    """Test the cos_sin_mean_wave_direction filter."""
    filter = filter_registry.create(
        "cos_sin_mean_wave_direction",
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    )
    pipeline = mwd_source | filter

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"cos_mwd", "sin_mwd"}
    assert len(output_fields["cos_mwd"]) == 1
    assert len(output_fields["sin_mwd"]) == 1

    assert np.allclose(output_fields["cos_mwd"][0].to_numpy(), COS_MWD_VALUES)
    assert np.allclose(output_fields["sin_mwd"][0].to_numpy(), SIN_MWD_VALUES)


def test_cos_sin_mean_wave_direction_reverse(cos_sin_mwd_source):
    """Test the cos_sin_mean_wave_direction filter in reverse."""
    filter = filter_registry.create(
        "cos_sin_mean_wave_direction",
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    ).reverse()
    pipeline = cos_sin_mwd_source | filter

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"mwd"}
    assert len(output_fields["mwd"]) == 1

    assert np.allclose(output_fields["mwd"][0].to_numpy(), MWD_VALUES)


def test_cos_sin_mean_wave_direction_round_trip(mwd_source):
    """Test the cos_sin_mean_wave_direction filter reproduces inputs on a round trip."""
    filter = filter_registry.create(
        "cos_sin_mean_wave_direction",
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    )
    cos_sin_mwd_source = mwd_source | filter
    pipeline = cos_sin_mwd_source | filter.reverse()

    input_fields = collect_fields_by_param(mwd_source)
    intermediate_fields = collect_fields_by_param(cos_sin_mwd_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"mwd"}
    assert set(intermediate_fields) == {"cos_mwd", "sin_mwd"}
    assert set(output_fields) == {"mwd"}

    for input_field, output_field in zip(input_fields["mwd"], output_fields["mwd"]):
        assert_fields_equal(input_field, output_field)


if __name__ == "__main__":
    """
    Run all test functions that start with 'test_'.
    """
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
