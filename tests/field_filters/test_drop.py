# (C) Copyright 2026 Anemoi contributors.
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

from ..utils import collect_fields_by_param

INPUT_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 30.0, 40.0],
    "valid_datetime": "2018-08-01T12:00:00Z",
}

MOCK_VALUES = np.array([1.0, 2.0, 3.0])


@pytest.fixture
def source(test_source):
    FIELD_SPECS = [
        {"param": "t", "levelist": 500, "values": MOCK_VALUES.copy(), **INPUT_METADATA},
        {
            "param": "t",
            "levelist": 850,
            "values": MOCK_VALUES.copy() * 2,
            **INPUT_METADATA,
        },
        {
            "param": "z",
            "levelist": 500,
            "values": MOCK_VALUES.copy() * 3,
            **INPUT_METADATA,
        },
        {
            "param": "z",
            "levelist": 850,
            "values": MOCK_VALUES.copy() * 4,
            **INPUT_METADATA,
        },
    ]
    return test_source(FIELD_SPECS)


def test_drop_by_param(source):
    drop = create_filter("drop", param="t")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "t" not in output_fields
    assert "z" in output_fields
    assert len(output_fields["z"]) == 2


def test_drop_by_levelist(source):
    drop = create_filter("drop", levelist=500)
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "t" in output_fields
    assert "z" in output_fields
    assert len(output_fields["t"]) == 1
    assert len(output_fields["z"]) == 1

    assert output_fields["t"][0].metadata("levelist") == 850
    assert output_fields["z"][0].metadata("levelist") == 850


def test_drop_by_param_and_levelist(source):
    drop = create_filter("drop", param="t", levelist=850)
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "t" in output_fields
    assert "z" in output_fields
    assert len(output_fields["t"]) == 1
    assert len(output_fields["z"]) == 2

    assert output_fields["t"][0].metadata("levelist") == 500


def test_drop_preserves_values(source):
    drop = create_filter("drop", param="t")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    for field in output_fields["z"]:
        level = field.metadata("levelist")
        if level == 500:
            np.testing.assert_array_equal(field.to_numpy(flatten=True), MOCK_VALUES * 3)
        elif level == 850:
            np.testing.assert_array_equal(field.to_numpy(flatten=True), MOCK_VALUES * 4)


def test_drop_no_match(source):
    drop = create_filter("drop", param="q")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)

    assert "t" in output_fields
    assert "z" in output_fields
    assert len(output_fields["t"]) == 2
    assert len(output_fields["z"]) == 2


def test_drop_all_fields(test_source):
    FIELD_SPECS = [
        {"param": "t", "values": MOCK_VALUES.copy(), **INPUT_METADATA},
        {"param": "t", "values": MOCK_VALUES.copy() * 2, **INPUT_METADATA},
    ]
    source = test_source(FIELD_SPECS)
    drop = create_filter("drop", param="t")
    pipeline = source | drop

    output_fields = collect_fields_by_param(pipeline)
    assert len(output_fields) == 0
