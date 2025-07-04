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


EXPECTED_SUM = {
    850: np.array([[331.23392773, 363.7319746], [356.15775585, 362.72220898], [349.92924023, 338.56595898]]),
    1000: np.array([[374.10890579, 380.71632767], [339.5542183, 364.07570267], [376.85304642, 349.41359329]]),
}


@pytest.fixture
def sum_input_source(test_source):
    PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"param": "r", "levelist": 850, "values": R_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 850, "values": T_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "q", "levelist": 850, "values": Q_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "r", "levelist": 1000, "values": R_VALUES[1000], **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 1000, "values": T_VALUES[1000], **MOCK_FIELD_METADATA},
        {"param": "q", "levelist": 1000, "values": Q_VALUES[1000], **MOCK_FIELD_METADATA},
    ]
    return test_source(PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def sum_output_source(test_source):
    PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"param": "sum", "levelist": 850, "values": R_VALUES[850], **MOCK_FIELD_METADATA},
        {"param": "sum", "levelist": 850, "values": T_VALUES[850], **MOCK_FIELD_METADATA},
    ]
    return test_source(PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC)


@skip_if_offline
def test_sum_fields(sum_input_source):

    sum_filter = filter_registry.create("sum", params=["r", "t"], output="sum")
    pipeline = sum_input_source | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    # Check the output contains the sum field and original inputs
    assert set(output_fields) == {"q", "sum"}

    # Check there is only one field as output
    assert len(output_fields["sum"]) == 2

    # Validate the sum field
    assert output_fields["sum"][0].to_numpy().shape == R_VALUES[850].shape
    assert np.allclose(output_fields["sum"][0].to_numpy(), EXPECTED_SUM[850])
    assert np.allclose(output_fields["sum"][1].to_numpy(), EXPECTED_SUM[1000])


def test_sum_filter_backward_not_implemented(sum_input_source):
    sum_filter = filter_registry.create("sum", params=["r", "t"], output="sum")

    pipeline = sum_input_source | sum_filter
    reverse = pipeline | sum_filter

    # Try calling backward_transform and confirm it raises NotImplementedError
    with pytest.raises(NotImplementedError):
        list(sum_filter.backward(reverse))


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
