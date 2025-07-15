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

from anemoi.transform.constants import g_gravitational_acceleration
from anemoi.transform.filters import filter_registry

from .utils import assert_fields_equal
from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

OROG_VALUES = np.array([[243.87788459, 1892.45371246], [427.80215359, 156.92873391], [2167.93458212, 338.15794671]])

Z_VALUES = OROG_VALUES * g_gravitational_acceleration


@pytest.fixture
def orog_source(test_source):
    OROG_SPEC = [{"param": "orog", "values": OROG_VALUES, **MOCK_FIELD_METADATA}]
    return test_source(OROG_SPEC)


@pytest.fixture
def z_source(test_source):
    Z_SPEC = [{"param": "z", "values": Z_VALUES, **MOCK_FIELD_METADATA}]
    return test_source(Z_SPEC)


def test_orog_to_z(orog_source):
    orog_to_z = filter_registry.create("orog_to_z")
    pipeline = orog_source | orog_to_z

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"z"}
    assert len(output_fields["z"]) == 1
    assert np.allclose(output_fields["z"][0].to_numpy(), Z_VALUES)


def test_orog_to_z_round_trip(orog_source):
    orog_to_z = filter_registry.create("orog_to_z")
    z_to_orog = filter_registry.create("z_to_orog")
    z_source = orog_source | orog_to_z
    pipeline = z_source | z_to_orog

    input_fields = collect_fields_by_param(orog_source)
    intermediate_fields = collect_fields_by_param(z_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"orog"}
    assert len(input_fields["orog"]) == 1
    assert set(intermediate_fields) == {"z"}
    assert len(intermediate_fields["z"]) == 1
    assert set(output_fields) == {"orog"}
    assert len(output_fields["orog"]) == 1
    assert_fields_equal(input_fields["orog"][0], output_fields["orog"][0])


def test_z_to_orog(z_source):
    z_to_orog = filter_registry.create("z_to_orog")
    pipeline = z_source | z_to_orog

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"orog"}
    assert len(output_fields["orog"]) == 1
    assert np.allclose(output_fields["orog"][0].to_numpy(), OROG_VALUES)


def test_z_to_orog_round_trip(z_source):
    z_to_orog = filter_registry.create("z_to_orog")
    orog_to_z = filter_registry.create("orog_to_z")
    orog_source = z_source | z_to_orog
    pipeline = orog_source | orog_to_z

    input_fields = collect_fields_by_param(z_source)
    intermediate_fields = collect_fields_by_param(orog_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"z"}
    assert len(input_fields["z"]) == 1
    assert set(intermediate_fields) == {"orog"}
    assert len(intermediate_fields["orog"]) == 1
    assert set(output_fields) == {"z"}
    assert len(output_fields["z"]) == 1
    assert_fields_equal(input_fields["z"][0], output_fields["z"][0])


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
