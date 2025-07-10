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
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry

from .utils import SelectFieldSource
from .utils import assert_fields_equal
from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}
R_VALUES = np.array([[78.13834333, 71.28598853], [99.17328572, 44.52144788], [56.49667261, 86.10495618]])

T_VALUES = np.array([[298.42488098, 297.55574036], [278.68269348, 293.99324036], [300.61042786, 300.40144348]])

D_VALUES = np.array([[294.34245300, 292.02214050], [278.56315613, 281.47135925], [291.19792175, 297.87370300]])


@pytest.fixture
def relative_humidity_source(test_source):
    RELATIVE_HUMIDITY_SPEC = [
        {"param": "r", "values": R_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "values": T_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def dewpoint_source(test_source):
    DEWPOINT_SPEC = [
        {"param": "d", "values": D_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "values": T_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(DEWPOINT_SPEC)


def test_relative_humidity_to_dewpoint(relative_humidity_source):
    r_to_d = filter_registry.create("r_to_d")
    pipeline = relative_humidity_source | r_to_d

    input_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"r", "t"}
    assert set(output_fields) == {"r", "t", "d"}

    # test unchanged fields agree
    for param in ("r", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test new output matches expected values
    assert len(output_fields["d"]) == 1
    result = output_fields["d"][0]
    expected_dewpoint = D_VALUES
    assert np.allclose(result.to_numpy(), expected_dewpoint)


def test_relative_humidity_to_dewpoint_round_trip(relative_humidity_source):
    r_to_d = filter_registry.create("r_to_d")
    d_to_r = filter_registry.create("d_to_r")
    # drop r to be sure it is reconstructed properly
    dewpoint_source = SelectFieldSource(relative_humidity_source | r_to_d, params=["t", "d"])
    pipeline = dewpoint_source | d_to_r

    input_fields = collect_fields_by_param(relative_humidity_source)
    intermediate_fields = collect_fields_by_param(dewpoint_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"r", "t"}
    assert set(intermediate_fields) == {"d", "t"}
    assert set(output_fields) == {"r", "d", "t"}

    # test unchanged fields agree from beginning to end
    for param in ("r", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("d", "t"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_relative_humidity_to_dewpoint_from_file(test_source):
    # this grib file is CERRA data that contains 2t and 2r
    source = test_source("anemoi-transform/filters/cerra_20240601_single_level.grib")
    r_to_d = filter_registry.create("r_to_d", relative_humidity="2r", temperature="2t", dewpoint="2d")
    pipeline = source | r_to_d

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2t", "2r"}
    assert set(output_fields) == {"2t", "2r", "2d"}

    # test unchanged fields agree
    for param in ("2r", "2t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    expected_dewpoint = test_source("anemoi-transform/filters/cerra_2d.npy").ds.to_numpy().reshape(1069, 1069)
    assert len(output_fields["2d"]) == 1
    result = output_fields["2d"][0]
    assert np.allclose(result.to_numpy(), expected_dewpoint)


def test_dewpoint_to_relative_humidity(dewpoint_source):
    d_to_r = filter_registry.create("d_to_r")
    pipeline = dewpoint_source | d_to_r

    input_fields = collect_fields_by_param(dewpoint_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"d", "t"}
    assert set(output_fields) == {"d", "t", "r"}

    # test unchanged fields agree
    for param in ("d", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test new output matches expected values
    assert len(output_fields["r"]) == 1
    result = output_fields["r"][0]
    expected_humidity = R_VALUES
    assert np.allclose(result.to_numpy(), expected_humidity)


def test_dewpoint_to_relative_humidity_round_trip(dewpoint_source):
    d_to_r = filter_registry.create("d_to_r")
    r_to_d = filter_registry.create("r_to_d")
    # drop d to be sure it is reconstructed properly
    relative_humidity_source = SelectFieldSource(dewpoint_source | d_to_r, params=["t", "r"])
    pipeline = relative_humidity_source | r_to_d

    input_fields = collect_fields_by_param(dewpoint_source)
    intermediate_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"d", "t"}
    assert set(intermediate_fields) == {"r", "t"}
    assert set(output_fields) == {"r", "d", "t"}

    # test unchanged fields agree from beginning to end
    for param in ("d", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("r", "t"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


def test_dewpoint_to_relative_humidity_from_file(test_source):
    dewpoint_source = test_source("anemoi-transform/filters/era_20240601_single_level_dewpoint.grib")
    d_to_r = filter_registry.create("d_to_r", relative_humidity="2r", temperature="2t", dewpoint="2d")
    pipeline = dewpoint_source | d_to_r

    input_fields = collect_fields_by_param(dewpoint_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2t", "2d"}
    assert set(output_fields) == {"2t", "2d", "2r"}

    # test unchanged fields agree
    for param in ("2t", "2d"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    expected_relative_humidity = test_source("anemoi-transform/filters/era5_2r.npy").ds.to_numpy().reshape(9, 18)
    assert len(output_fields["2r"]) == 1
    result = output_fields["2r"][0]
    assert np.allclose(result.to_numpy(), expected_relative_humidity)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
