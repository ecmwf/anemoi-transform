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
from anemoi.utils.testing import skip_if_offline
from earthkit.data.indexing.fieldlist import SimpleFieldList
from earthkit.data.sources.array_list import ArrayField
from earthkit.data.utils.metadata.dict import UserMetadata

from anemoi.transform.filters import create_filter_by_name as create_filter
from anemoi.transform.sources import source_registry

from ..utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

T_VALUES = np.array([[293.32301331, 284.21559143], [260.53981018, 291.18824768], [279.88941956, 248.87574768]])

Q_VALUES = np.array([[0.00657578, 0.00769957], [0.00147607, 0.01088967], [0.00505508, 0.00044559]])

R_VALUES = np.array([[37.91091442, 79.51638317], [95.61794567, 71.53396130], [70.03982067, 89.69021130]])

EXPECTED_SUM = (R_VALUES + T_VALUES).flatten()

EXPECTED_SUM_MULTILEVEL = (T_VALUES * 2.0 - 15.0).flatten()


@pytest.fixture
def sum_input_source_one_level(mars_test_source):
    PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC = [
        {"param": "r", "levelist": 850, "values": R_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 850, "values": T_VALUES, **MOCK_FIELD_METADATA},
        {"param": "q", "levelist": 850, "values": Q_VALUES, **MOCK_FIELD_METADATA},
    ]
    return mars_test_source(PRESSURE_LEVEL_RELATIVE_HUMIDITY_SPEC)


@pytest.fixture
def sum_input_source_multilevel(mars_test_source):
    MULTILEVEL_TEMP_RELATIVE_HUMIDITY = [
        {"param": "t_850", "levelist": 850, "values": T_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t_500", "levelist": 500, "values": T_VALUES - 15.0, **MOCK_FIELD_METADATA},
        {"param": "r", "levelist": 850, "values": R_VALUES, **MOCK_FIELD_METADATA},
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
    assert output_fields["sum"][0].to_numpy().shape == EXPECTED_SUM_MULTILEVEL.shape
    assert np.allclose(output_fields["sum"][0].to_numpy(), EXPECTED_SUM_MULTILEVEL)


def test_sum_multilevel_ignore_level_false(sum_input_source_multilevel):
    sum_filter = create_filter("sum", params=["t_850", "t_500"], output="sum")
    pipeline = sum_input_source_multilevel | sum_filter

    # Try calling sum of different levels without ignoring the level
    with pytest.raises(ValueError):
        list(sum_filter.forward(pipeline))


@pytest.fixture
def no_mars_test_source():
    """Source whose fields have generic metadata but NO mars namespace,
    mimicking netCDF / anemoi-datasets fields.
    """

    def _source(dataset: list[dict]):
        fields = []
        for d in dataset:
            v = np.array(d["values"])
            fields.append(ArrayField(v, UserMetadata(d, shape=v.shape)))
        return source_registry.create("testing", dataset=SimpleFieldList(fields=fields))

    return _source


@pytest.fixture
def sum_input_source_one_level_no_mars(no_mars_test_source):
    SPEC = [
        {"param": "r", "levelist": 850, "values": R_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t", "levelist": 850, "values": T_VALUES, **MOCK_FIELD_METADATA},
        {"param": "q", "levelist": 850, "values": Q_VALUES, **MOCK_FIELD_METADATA},
    ]
    return no_mars_test_source(SPEC)


@pytest.fixture
def sum_input_source_multilevel_no_mars(no_mars_test_source):
    SPEC = [
        {"param": "t_850", "levelist": 850, "values": T_VALUES, **MOCK_FIELD_METADATA},
        {"param": "t_500", "levelist": 500, "values": T_VALUES - 15.0, **MOCK_FIELD_METADATA},
        {"param": "r", "levelist": 850, "values": R_VALUES, **MOCK_FIELD_METADATA},
    ]
    return no_mars_test_source(SPEC)


def test_sum_fields_no_mars_namespace(sum_input_source_one_level_no_mars):
    """Sum must work when fields have no mars namespace (netCDF, anemoi-datasets)."""
    sum_filter = create_filter("sum", params=["r", "t"], output="sum")
    pipeline = sum_input_source_one_level_no_mars | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"q", "sum"}
    assert len(output_fields["sum"]) == 1
    assert output_fields["sum"][0].to_numpy().shape == EXPECTED_SUM.shape
    assert np.allclose(output_fields["sum"][0].to_numpy(), EXPECTED_SUM)


def test_sum_fields_multilevel_no_mars_namespace(sum_input_source_multilevel_no_mars):
    """ignore_level must also work without a mars namespace."""
    sum_filter = create_filter("sum", params=["t_850", "t_500"], output="sum", ignore_level=True)
    pipeline = sum_input_source_multilevel_no_mars | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"r", "sum"}
    assert len(output_fields["sum"]) == 1
    assert np.allclose(output_fields["sum"][0].to_numpy(), EXPECTED_SUM_MULTILEVEL)


def test_sum_no_mars_distinct_datetimes(no_mars_test_source):
    """Fields at different valid times must NOT be merged into one group."""
    spec = [
        {
            "param": "r",
            "values": R_VALUES,
            "valid_datetime": "2018-08-01T09:00:00Z",
            "latitudes": [10.0, 0.0, -10.0],
            "longitudes": [20, 40.0],
        },
        {
            "param": "t",
            "values": T_VALUES,
            "valid_datetime": "2018-08-01T09:00:00Z",
            "latitudes": [10.0, 0.0, -10.0],
            "longitudes": [20, 40.0],
        },
        {
            "param": "r",
            "values": R_VALUES,
            "valid_datetime": "2018-08-01T10:00:00Z",
            "latitudes": [10.0, 0.0, -10.0],
            "longitudes": [20, 40.0],
        },
        {
            "param": "t",
            "values": T_VALUES,
            "valid_datetime": "2018-08-01T10:00:00Z",
            "latitudes": [10.0, 0.0, -10.0],
            "longitudes": [20, 40.0],
        },
    ]
    source = no_mars_test_source(spec)
    sum_filter = create_filter("sum", params=["r", "t"], output="sum")
    pipeline = source | sum_filter
    output_fields = collect_fields_by_param(pipeline)

    # Two timesteps -> two sum fields, no duplicate-key crash
    assert len(output_fields["sum"]) == 2
    for f in output_fields["sum"]:
        assert np.allclose(f.to_numpy(), EXPECTED_SUM)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
