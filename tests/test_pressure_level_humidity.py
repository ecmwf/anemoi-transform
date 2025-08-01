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

T_VALUES = {
    850: np.array([[293.32301331, 284.21559143], [260.53981018, 291.18824768], [279.88941956, 248.87574768]]),
    1000: np.array([[291.22831726, 289.85136414], [271.29277039, 301.67362976], [287.53691101, 250.15409851]]),
}

Q_VALUES = {
    850: np.array([[0.00657578, 0.00769957], [0.00147607, 0.01088967], [0.00505508, 0.00044559]]),
    1000: np.array([[0.01075057, 0.01080445], [0.00226020, 0.01525551], [0.00914679, 0.00047560]]),
}

R_VALUES = {
    850: np.array([[37.91091442, 79.51638317], [95.61794567, 71.53396130], [70.03982067, 89.69021130]]),
    1000: np.array([[82.88058853, 90.86496353], [68.26144791, 62.40207291], [89.31613541, 99.25949478]]),
}


@pytest.fixture
def relative_humidity_source(test_source):
    PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"param": "r", "levelist": 850, "values": R_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 850, "values": T_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "r", "levelist": 1000, "values": R_VALUES[1000], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 1000, "values": T_VALUES[1000], **MOCK_FIELD_METADATA},
    ]
    return test_source(PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def specific_humidity_source(test_source):
    PRESSURE_LEVEL_SPECIFIC_HUMIDITY_SPEC = [
        {"param": "q", "levelist": 850, "values": Q_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 850, "values": T_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "q", "levelist": 1000, "values": Q_VALUES[1000], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 1000, "values": T_VALUES[1000], **MOCK_FIELD_METADATA},
    ]
    return test_source(PRESSURE_LEVEL_SPECIFIC_HUMIDITY_SPEC)


def test_pressure_level_specific_humidity_to_relative_humidity(specific_humidity_source):
    q_to_r = filter_registry.create("q_to_r")
    pipeline = specific_humidity_source | q_to_r

    input_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"q", "t"}
    assert set(output_fields) == {"q", "t", "r"}

    # test unchanged fields agree
    for param in ("q", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test new output matches expected values
    results_by_level = {field.metadata("levelist"): field.to_numpy() for field in output_fields["r"]}
    assert set(results_by_level) == {850, 1000}

    for level, result in results_by_level.items():
        expected_relative_humidity = R_VALUES[level]
        assert np.allclose(result, expected_relative_humidity)


def test_pressure_level_specific_humidity_to_relative_humidity_round_trip(specific_humidity_source):
    q_to_r = filter_registry.create("q_to_r")
    r_to_q = filter_registry.create("r_to_q")

    relative_humidity_source = SelectFieldSource(specific_humidity_source | q_to_r, params=["r", "t"])
    pipeline = relative_humidity_source | r_to_q

    input_fields = collect_fields_by_param(specific_humidity_source)
    intermediate_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"q", "t"}
    assert set(intermediate_fields) == {"r", "t"}
    assert set(output_fields) == {"q", "t", "r"}

    # check unchanged fields agree from beginning to end
    for param in ("q", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("r", "t"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_pressure_level_specific_humidity_to_relative_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/era_20240601_pressure_level_specific_humidity.grib")
    q_to_r = filter_registry.create("q_to_r")
    pipeline = source | q_to_r

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"q", "t"}
    assert set(output_fields) == {"q", "t", "r"}

    # test unchanged fields agree
    for param in ("q", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    fields = sorted(output_fields["r"], key=lambda f: f.metadata("levelist"))
    fields = map(lambda f: f.to_numpy(), fields)
    result = np.stack(list(fields)).flatten()

    expected_relative_humidity = test_source("anemoi-transform/filters/era_r.npy").ds.to_numpy().flatten()
    assert np.allclose(result, expected_relative_humidity)


def test_pressure_level_relative_humidity_to_specific_humidity(relative_humidity_source):
    r_to_q = filter_registry.create("r_to_q")
    pipeline = relative_humidity_source | r_to_q

    input_fields = collect_fields_by_param(relative_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"r", "t"}
    assert set(output_fields) == {"r", "t", "q"}

    # test unchanged fields agree
    for param in ("r", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test new output matches expected values
    results_by_level = {field.metadata("levelist"): field.to_numpy() for field in output_fields["q"]}
    assert set(results_by_level) == {850, 1000}

    for level, result in results_by_level.items():
        expected_specific_humidity = Q_VALUES[level]
        assert np.allclose(result, expected_specific_humidity)


def test_pressure_level_relative_humidity_to_specific_humidity_round_trip(relative_humidity_source):
    r_to_q = filter_registry.create("r_to_q")
    q_to_r = filter_registry.create("q_to_r")
    specific_humidity_source = SelectFieldSource(relative_humidity_source | r_to_q, params=["q", "t"])
    pipeline = specific_humidity_source | q_to_r

    input_fields = collect_fields_by_param(relative_humidity_source)
    intermediate_fields = collect_fields_by_param(specific_humidity_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"r", "t"}
    assert set(intermediate_fields) == {"q", "t"}
    assert set(output_fields) == {"r", "t", "q"}

    # test unchanged fields agree from beginning to end
    for param in ("r", "t"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("q", "t"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_pressure_level_relative_humidity_to_specific_humidity_from_file_arome(test_source):
    source = test_source("anemoi-transform/filters/r_t_PAAROME_1S40_ECH0_ISOBARE.grib")
    r_to_q = filter_registry.create("r_to_q")
    pipeline = source | r_to_q

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"t", "r"}
    assert set(output_fields) == {"q", "t", "r"}

    # test unchanged fields agree
    for param in ("t", "r"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    fields = sorted(output_fields["q"], key=lambda f: f.metadata("levelist"))
    fields = map(lambda f: f.to_numpy(), fields)
    result = np.stack(list(fields))
    result = result.flatten()

    expected_specific_humidity = (
        test_source("anemoi-transform/filters/arome_specific_humidity.npy").ds.to_numpy().flatten()
    )

    assert np.allclose(result, expected_specific_humidity, equal_nan=True)


@skip_if_offline
def test_pressure_level_relative_humidity_to_specific_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/cerra_20240601_pressure_levels.grib")
    r_to_q = filter_registry.create("r_to_q")
    pipeline = source | r_to_q

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"t", "r"}
    assert set(output_fields) == {"q", "t", "r"}

    # test unchanged fields agree
    for param in ("t", "r"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test pipeline output matches known good output
    fields = sorted(output_fields["q"], key=lambda f: f.metadata("levelist"))
    fields = map(lambda f: f.to_numpy(), fields)
    result = np.stack(list(fields)).flatten()

    expected_specific_humidity = test_source("anemoi-transform/filters/cerra_q.npy").ds.to_numpy().flatten()
    assert np.allclose(result, expected_specific_humidity)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
