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

from anemoi.transform.filters import create_filter_by_name as create_filter

from ..utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "geography.distinct_latitudes": [10.0, 0.0, -10.0],
    "geography.distinct_longitudes": [20, 40.0],
    "time.valid_datetime": "2018-08-01T09:00:00Z",
}

T_VALUES = np.array([[293.32301331, 284.21559143], [260.53981018, 291.18824768], [279.88941956, 248.87574768]])

Q_VALUES = np.array([[0.00657578, 0.00769957], [0.00147607, 0.01088967], [0.00505508, 0.00044559]])

R_VALUES = np.array([[37.91091442, 79.51638317], [95.61794567, 71.53396130], [70.03982067, 89.69021130]])

EXPECTED_SUM = R_VALUES + T_VALUES

EXPECTED_SUM_MULTILEVEL = (T_VALUES * 2.0 - 15.0).flatten()


@pytest.fixture
def sum_input_source_one_level(mars_test_source):
    PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"parameter.variable": "r", "vertical.level": 850, "data.values": R_VALUES, **MOCK_FIELD_METADATA},
        {"parameter.variable": "t", "vertical.level": 850, "data.values": T_VALUES, **MOCK_FIELD_METADATA},
        {"parameter.variable": "q", "vertical.level": 850, "data.values": Q_VALUES, **MOCK_FIELD_METADATA},
    ]
    return mars_test_source(PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def sum_input_source_multilevel(mars_test_source):
    MULTILEVEL_TEMP_RELATIVE_HUMIDITY = [
        {"parameter.variable": "t_850", "vertical.level": 850, "data.values": T_VALUES, **MOCK_FIELD_METADATA},
        {"parameter.variable": "t_500", "vertical.level": 500, "data.values": T_VALUES - 15.0, **MOCK_FIELD_METADATA},
        {"parameter.variable": "r", "vertical.level": 850, "data.values": R_VALUES, **MOCK_FIELD_METADATA},
    ]
    return mars_test_source(MULTILEVEL_TEMP_RELATIVE_HUMIDITY)


@skip_if_offline
def test_sum_fields(sum_input_source_one_level):

    sum_filter = create_filter("sum", params=["r", "t"], output="sum")
    pipeline = sum_input_source_one_level | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    # Check the output contains the sum field and original inputs
    assert set(output_fields) == {"q", "sum"}

    # Check there is only one field as output
    assert len(output_fields["sum"]) == 1

    # Validate the sum field
    # arrays are flattened in sum
    assert output_fields["sum"][0].to_numpy().shape == EXPECTED_SUM.shape
    assert np.allclose(output_fields["sum"][0].to_numpy(), EXPECTED_SUM)


def test_sum_filter_backward_not_implemented(sum_input_source_one_level):
    sum_filter = create_filter("sum", params=["r", "t"], output="sum")

    pipeline = sum_input_source_one_level | sum_filter
    reverse = pipeline | sum_filter

    # Try calling backward_transform and confirm it raises NotImplementedError
    with pytest.raises(NotImplementedError):
        list(sum_filter.backward(reverse))


def test_sum_fields_multilevel(sum_input_source_multilevel):
    sum_filter = create_filter("sum", params=["t_850", "t_500"], output="sum", ignore_level=True)

    pipeline = sum_input_source_multilevel | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    # Check the output contains the sum field and original inputs
    assert set(output_fields) == {"r", "sum"}

    # Check there is only one field as output
    assert len(output_fields["sum"]) == 1

    # Validate the sum field
    # arrays are flattened in sum
    assert output_fields["sum"][0].to_numpy(flatten=True).shape == EXPECTED_SUM_MULTILEVEL.shape
    assert np.allclose(output_fields["sum"][0].to_numpy(flatten=True), EXPECTED_SUM_MULTILEVEL)


def test_sum_multilevel_ignore_level_false(sum_input_source_multilevel):
    sum_filter = create_filter("sum", params=["t_850", "t_500"], output="sum")
    pipeline = sum_input_source_multilevel | sum_filter

    # Try calling sum of different levels without ignoring the level
    with pytest.raises(ValueError):
        list(sum_filter.forward(pipeline))


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
