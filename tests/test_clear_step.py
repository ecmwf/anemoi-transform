# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest
from earthkit.data.utils.dates import to_datetime

from anemoi.transform.filters import filter_registry

from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T12:00:00Z",
}

MOCK_VALUES = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def source(test_source):
    FIELD_SPECS = [
        {"param": "t", "step": 0, "values": MOCK_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "step": 6, "values": MOCK_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "step": 12, "values": MOCK_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(FIELD_SPECS)


def test_clear_step(source):
    clear_step = filter_registry.create("clear_step")
    pipeline = source | clear_step

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    param = "t"
    assert set(input_fields) == {param}
    assert len(input_fields[param]) == 3
    assert set(output_fields) == {param}
    assert len(output_fields[param]) == 3

    for input_field, output_field in zip(input_fields[param], output_fields[param]):
        input_validtime = to_datetime(input_field.metadata("valid_datetime"))
        output_validtime = to_datetime(output_field.metadata("valid_datetime"))
        input_step = input_field.metadata("step")

        expected_validtime = input_validtime - datetime.timedelta(hours=input_step)
        assert output_validtime == expected_validtime
        assert output_field.metadata("step") == 0

        assert np.array_equal(input_field.to_numpy(), output_field.to_numpy())


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
