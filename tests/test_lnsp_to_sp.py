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

LNSP_VALUES = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

SP_VALUES = np.exp(LNSP_VALUES)


@pytest.fixture
def lnsp_source(test_source):
    LNSP_SPEC = [{"param": "lnsp", "values": LNSP_VALUES, **MOCK_FIELD_METADATA}]
    return test_source(LNSP_SPEC)


@pytest.fixture
def sp_source(test_source):
    SP_SPEC = [{"param": "sp", "values": SP_VALUES, **MOCK_FIELD_METADATA}]
    return test_source(SP_SPEC)


def test_lnsp_to_sp(lnsp_source):
    lnsp_to_sp = filter_registry.create("lnsp_to_sp")
    pipeline = lnsp_source | lnsp_to_sp

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"sp"}
    assert len(output_fields["sp"]) == 1
    assert np.allclose(output_fields["sp"][0].to_numpy(), SP_VALUES)


def test_lnsp_to_sp_round_trip(lnsp_source):
    lnsp_to_sp = filter_registry.create("lnsp_to_sp")
    sp_to_lnsp = filter_registry.create("sp_to_lnsp")
    sp_source = lnsp_source | lnsp_to_sp
    pipeline = sp_source | sp_to_lnsp

    input_fields = collect_fields_by_param(lnsp_source)
    intermediate_fields = collect_fields_by_param(sp_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"lnsp"}
    assert len(input_fields["lnsp"]) == 1
    assert set(intermediate_fields) == {"sp"}
    assert len(intermediate_fields["sp"]) == 1
    assert set(output_fields) == {"lnsp"}
    assert len(output_fields["lnsp"]) == 1
    assert_fields_equal(input_fields["lnsp"][0], output_fields["lnsp"][0], exclude_keys=["levelist"])


def test_sp_to_lnsp(sp_source):
    sp_to_lnsp = filter_registry.create("sp_to_lnsp")
    pipeline = sp_source | sp_to_lnsp

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"lnsp"}
    assert len(output_fields["lnsp"]) == 1
    assert np.allclose(output_fields["lnsp"][0].to_numpy(), LNSP_VALUES)


def test_sp_to_lnsp_round_trip(sp_source):
    sp_to_lnsp = filter_registry.create("sp_to_lnsp")
    lnsp_to_sp = filter_registry.create("lnsp_to_sp")
    lnsp_source = sp_source | sp_to_lnsp
    pipeline = lnsp_source | lnsp_to_sp

    input_fields = collect_fields_by_param(sp_source)
    intermediate_fields = collect_fields_by_param(lnsp_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"sp"}
    assert len(input_fields["sp"]) == 1
    assert set(intermediate_fields) == {"lnsp"}
    assert len(intermediate_fields["lnsp"]) == 1
    assert set(output_fields) == {"sp"}
    assert len(output_fields["sp"]) == 1
    assert_fields_equal(input_fields["sp"][0], output_fields["sp"][0], exclude_keys=["levelist"])
