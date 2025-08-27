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

U_VALUES = {
    500: np.array([[-3.26786804, -2.90458679], [-4.28153992, -10.75224304], [-6.29130554, -4.17704773]]),
    850: np.array([[-6.72481718, -0.34174164], [-7.14725727, -2.02047454], [-4.93597360, -0.00018431]]),
}

V_VALUES = {
    500: np.array([[6.51824951, 4.7321167], [1.16961670, 1.73797607], [-2.93096924, 3.2399292]]),
    850: np.array([[5.4374572, -0.00679462], [2.23226754, 6.78457592], [-1.79188286, -0.0093771]]),
}

DD_VALUES = {
    500: np.array([[7.29153881, 5.55243666], [4.43842171, 10.89179926], [6.94054076, 5.28629066]]),
    850: np.array([[8.64806955, 0.34180918], [7.48774364, 7.07903862], [5.25115983, 0.00937891]]),
}

FF_VALUES = {
    500: np.array([[153.37349864, 148.45827835], [105.27908047, 99.18178736], [65.02031089, 127.79896253]]),
    850: np.array([[128.95781905, 88.86097611], [107.34489249, 163.41625261], [70.04782648, 1.12603196]]),
}


@pytest.fixture
def uv_source(test_source):
    UV_SPEC = [
        {"param": "u", "levelist": 500, "values": U_VALUES[500], **MOCK_FIELD_METADATA},
        {"param": "v", "levelist": 500, "values": V_VALUES[500], **MOCK_FIELD_METADATA},
        {"param": "u", "levelist": 850, "values": U_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "v", "levelist": 850, "values": V_VALUES[850], **MOCK_FIELD_METADATA},
    ]
    return test_source(UV_SPEC)


@pytest.fixture
def ddff_source(test_source):
    DDFF_SPEC = [
        {"param": "ws", "levelist": 500, "values": DD_VALUES[500], **MOCK_FIELD_METADATA},
        {"param": "wdir", "levelist": 500, "values": FF_VALUES[500], **MOCK_FIELD_METADATA},
        {"param": "ws", "levelist": 850, "values": DD_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "wdir", "levelist": 850, "values": FF_VALUES[850], **MOCK_FIELD_METADATA},
    ]
    return test_source(DDFF_SPEC)


def test_uv_to_ddff(uv_source):
    uv_to_ddff = filter_registry.create("uv_to_ddff")
    pipeline = uv_source | uv_to_ddff

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"ws", "wdir"}
    assert len(output_fields["ws"]) == 2
    assert len(output_fields["wdir"]) == 2
    for i, level in enumerate([500, 850]):
        assert np.allclose(output_fields["ws"][i].to_numpy(), DD_VALUES[level])
        assert np.allclose(output_fields["wdir"][i].to_numpy(), FF_VALUES[level])
        assert output_fields["ws"][i].metadata("levelist") == level
        assert output_fields["wdir"][i].metadata("levelist") == level


def test_uv_to_ddff_round_trip(uv_source):
    uv_to_ddff = filter_registry.create("uv_to_ddff")
    ddff_to_uv = filter_registry.create("ddff_to_uv")
    ddff_source = uv_source | uv_to_ddff
    pipeline = ddff_source | ddff_to_uv

    input_fields = collect_fields_by_param(uv_source)
    intermediate_fields = collect_fields_by_param(ddff_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"u", "v"}
    assert set(intermediate_fields) == {"ws", "wdir"}
    assert set(output_fields) == {"u", "v"}

    for param in ("u", "v"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)


def test_ddff_to_uv(ddff_source):
    ddff_to_uv = filter_registry.create("ddff_to_uv")
    pipeline = ddff_source | ddff_to_uv

    output_fields = collect_fields_by_param(pipeline)
    assert set(output_fields) == {"u", "v"}
    assert len(output_fields["u"]) == 2
    assert len(output_fields["v"]) == 2
    for i, level in enumerate([500, 850]):
        assert np.allclose(output_fields["u"][i].to_numpy(), U_VALUES[level])
        assert np.allclose(output_fields["v"][i].to_numpy(), V_VALUES[level])
        assert output_fields["u"][i].metadata("levelist") == level
        assert output_fields["v"][i].metadata("levelist") == level


def test_ddff_to_uv_round_trip(ddff_source):
    ddff_to_uv = filter_registry.create("ddff_to_uv")
    uv_to_ddff = filter_registry.create("uv_to_ddff")
    uv_source = ddff_source | ddff_to_uv
    pipeline = uv_source | uv_to_ddff

    input_fields = collect_fields_by_param(ddff_source)
    intermediate_fields = collect_fields_by_param(uv_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"ws", "wdir"}
    assert set(intermediate_fields) == {"u", "v"}
    assert set(output_fields) == {"ws", "wdir"}

    for param in ("ws", "wdir"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
