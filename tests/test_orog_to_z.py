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
OROG_VALUES = np.array([[0.78, 0.71], [0.99, 0.44], [0.56, 0.86]])

Z_VALUES = np.array([[7.649187,6.9627215], [9.7085835,4.314926], [5.491724, 8.433719]])


@pytest.fixture
def orography_source(test_source):
    OROGRAPHY_SPEC = [
        {"param": "orog", "values": OROG_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(OROGRAPHY_SPEC)


@pytest.fixture
def geopotential_source(test_source):
    Z_SPEC = [
        {"param": "z", "values": Z_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(Z_SPEC)


def test_orog_to_z(orography_source):
    orog_to_z = filter_registry.create("orog_to_z")
    pipeline = orography_source | orog_to_z
    input_fields = collect_fields_by_param(orography_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"orog"}
    assert set(output_fields) == {"z",'orog'}

    # test unchanged fields agree
    for param in ("orog"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test new output matches expected values
    assert len(output_fields["z"]) == 1
    result = output_fields["z"][0]
    expected_geopotential = Z_VALUES
    assert np.allclose(result.to_numpy(), expected_geopotential)


def test_orog_to_z_round_trip(orography_source):
    orog_to_z = filter_registry.create("orog_to_z")
    z_to_orog = filter_registry.create("z_to_orog")
    # drop orog to be sure it is reconstructed properly
    z_source = SelectFieldSource(orography_source | orog_to_z, params=["z"])
    pipeline = z_source | z_to_orog

    input_fields = collect_fields_by_param(orography_source)
    intermediate_fields = collect_fields_by_param(z_source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"orog"}
    assert set(intermediate_fields) == {"z"}
    assert set(output_fields) == {"z",'orog'}

    # test unchanged fields agree from beginning to end
    for param in ("orog"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)

    # test intermediate fields are unchanged
    for param in ("z"):
        for intermediate_field, output_field in zip(intermediate_fields[param], output_fields[param]):
            assert_fields_equal(intermediate_field, output_field)


@skip_if_offline
def test_orog_to_z_from_file(test_source):
    # this grib file is CARRA data that contains orog
    source = test_source("anemoi-transform/filters/carra_orography.grib")
    orog_to_z = filter_registry.create("orog_to_z")
    pipeline = source | orog_to_z

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"orog"}
    assert set(output_fields) == {"z","orog"}

    # test pipeline output matches known good output
    expected_geopotential = test_source("anemoi-transform/filters/carra_geopotential.npy").ds.to_numpy()
    assert len(output_fields["z"]) == 1
    result = output_fields["z"][0]
    print(result)
    assert np.allclose(result.to_numpy(), expected_geopotential)

if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
