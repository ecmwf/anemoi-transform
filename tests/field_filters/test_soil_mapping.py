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
from anemoi.transform.filters.fields.soil_mapping import SOIL_LEVTYPE
from anemoi.transform.filters.fields.soil_mapping import SoilMapping

from ..utils import assert_fields_equal
from ..utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

STL1_VALUES = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
STL2_VALUES = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
STL3_VALUES = np.array([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]])
SWVL1_VALUES = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
SWVL2_VALUES = np.array([[0.11, 0.22], [0.33, 0.44], [0.55, 0.66]])
SWVL3_VALUES = np.array([[0.12, 0.23], [0.34, 0.45], [0.56, 0.67]])
T2M_VALUES = np.array([[280.0, 281.0], [282.0, 283.0], [284.0, 285.0]])

# Mapping of named soil parameter to (generic param, soil level, values).
SOIL_CASES = {
    "stl1": ("sot", 1, STL1_VALUES),
    "stl2": ("sot", 2, STL2_VALUES),
    "stl3": ("sot", 3, STL3_VALUES),
    "swvl1": ("vsw", 1, SWVL1_VALUES),
    "swvl2": ("vsw", 2, SWVL2_VALUES),
    "swvl3": ("vsw", 3, SWVL3_VALUES),
}


@pytest.fixture
def soil_source(test_source):
    """A source of the named soil parameters (no levtype/levelist)."""
    spec = [{"param": name, "values": values, **MOCK_FIELD_METADATA} for name, (_, _, values) in SOIL_CASES.items()]
    return test_source(spec)


@pytest.fixture
def generic_soil_source(test_source):
    """A source of the generic soil parameters (with levtype/levelist)."""
    spec = [
        {
            "param": generic,
            "levtype": SOIL_LEVTYPE,
            "levelist": level,
            "values": values,
            **MOCK_FIELD_METADATA,
        }
        for (generic, level, values) in SOIL_CASES.values()
    ]
    return test_source(spec)


@pytest.fixture
def mixed_source(test_source):
    """A source mixing named soil parameters with an unrelated surface field."""
    spec = [
        {"param": "stl1", "values": STL1_VALUES, **MOCK_FIELD_METADATA},
        {"param": "swvl2", "values": SWVL2_VALUES, **MOCK_FIELD_METADATA},
        {"param": "2t", "values": T2M_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(spec)


def test_soil_to_levtype_sol(soil_source):
    soil_to_levtype = create_filter("soil_to_levtype_sol")
    pipeline = soil_source | soil_to_levtype

    output_fields = collect_fields_by_param(pipeline)

    # All named soil params become the two generic params.
    assert set(output_fields) == {"sot", "vsw"}
    assert len(output_fields["sot"]) == 3
    assert len(output_fields["vsw"]) == 3

    for fields, expected in (
        (output_fields["sot"], "sot"),
        (output_fields["vsw"], "vsw"),
    ):
        for field in fields:
            assert field.metadata("param") == expected
            assert field.metadata("levtype") == SOIL_LEVTYPE
            assert field.metadata("levelist") in (1, 2, 3)


def test_soil_to_levtype_sol_values_and_levels(soil_source):
    soil_to_levtype = create_filter("soil_to_levtype_sol")
    pipeline = soil_source | soil_to_levtype

    by_key = {(f.metadata("param"), f.metadata("levelist")): f for f in pipeline}

    for _, (generic, level, values) in SOIL_CASES.items():
        field = by_key[(generic, level)]
        assert np.allclose(field.to_numpy(), values)


def test_levtype_sol_to_soil(generic_soil_source):
    levtype_to_soil = create_filter("levtype_sol_to_soil")
    pipeline = generic_soil_source | levtype_to_soil

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == set(SOIL_CASES)
    for name, (_, _, values) in SOIL_CASES.items():
        assert len(output_fields[name]) == 1
        field = output_fields[name][0]
        assert np.allclose(field.to_numpy(), values)
        # levtype/levelist should be stripped from the named representation.
        assert field.metadata("levtype", default=None) is None
        assert field.metadata("levelist", default=None) is None


def test_soil_round_trip(soil_source):
    soil_to_levtype = create_filter("soil_to_levtype_sol")
    levtype_to_soil = create_filter("levtype_sol_to_soil")

    intermediate = soil_source | soil_to_levtype
    pipeline = intermediate | levtype_to_soil

    input_fields = collect_fields_by_param(soil_source)
    intermediate_fields = collect_fields_by_param(intermediate)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == set(SOIL_CASES)
    assert set(intermediate_fields) == {"sot", "vsw"}
    assert set(output_fields) == set(SOIL_CASES)

    for name in SOIL_CASES:
        assert_fields_equal(input_fields[name][0], output_fields[name][0], exclude_keys=["levelist"])


def test_levtype_sol_to_soil_round_trip(generic_soil_source):
    levtype_to_soil = create_filter("levtype_sol_to_soil")
    soil_to_levtype = create_filter("soil_to_levtype_sol")

    intermediate = generic_soil_source | levtype_to_soil
    pipeline = intermediate | soil_to_levtype

    input_fields = collect_fields_by_param(generic_soil_source)
    intermediate_fields = collect_fields_by_param(intermediate)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"sot", "vsw"}
    assert set(intermediate_fields) == set(SOIL_CASES)
    assert set(output_fields) == {"sot", "vsw"}

    input_by_key = {(f.metadata("param"), f.metadata("levelist")): f for f in generic_soil_source}
    output_by_key = {(f.metadata("param"), f.metadata("levelist")): f for f in pipeline}

    assert set(input_by_key) == set(output_by_key)
    for key, field in input_by_key.items():
        assert np.allclose(field.to_numpy(), output_by_key[key].to_numpy())


def test_non_soil_fields_pass_through(mixed_source):
    soil_to_levtype = create_filter("soil_to_levtype_sol")
    pipeline = mixed_source | soil_to_levtype

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"sot", "vsw", "2t"}
    # The unrelated field is unchanged.
    assert np.allclose(output_fields["2t"][0].to_numpy(), T2M_VALUES)
    assert output_fields["2t"][0].metadata("levtype", default=None) is None


def test_patch_data_request_replaces_named_params():
    filter = SoilMapping()
    request = {"param": ["stl1", "stl2", "swvl1", "2t"], "levtype": "sfc"}

    patched = filter.patch_data_request(request)

    assert set(patched["param"]) == {"sot", "vsw", "2t"}
    assert patched["levtype"] == SOIL_LEVTYPE
    assert patched["levelist"] == [1, 2]


def test_patch_data_request_merges_existing_levelist():
    filter = SoilMapping()
    request = {"param": ["stl3"], "levtype": "sfc", "levelist": [137]}

    patched = filter.patch_data_request(request)

    assert patched["param"] == ["sot"]
    assert patched["levtype"] == SOIL_LEVTYPE
    assert patched["levelist"] == [3, 137]


def test_patch_data_request_no_soil_params_unchanged():
    filter = SoilMapping()
    request = {"param": "2t", "levtype": "sfc"}

    patched = filter.patch_data_request(dict(request))

    assert patched == request


def test_patch_data_request_no_param_key():
    filter = SoilMapping()
    request = {"levtype": "sfc"}

    assert filter.patch_data_request(dict(request)) == request
