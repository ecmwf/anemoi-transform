# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from ..utils import collect_fields_by_param
from ..utils import create_dispatching_filter as create_filter

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

MASK_VALUES = {
    "all_zeros": np.array([[0, 0], [0, 0], [0, 0]]),
    "all_ones": np.array([[1, 1], [1, 1], [1, 1]]),
    "mixed_ints": np.array([[0, 1], [1, 0], [1, 2]]),
    "mixed_floats": np.array([[0.0, 0.25], [0.5, 0.5], [0.75, 1.0]]),
}

DATA_VALUES = {
    "t": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "q": np.array([[7.0, 8.0], [9.0, 0.0], [9.0, 8.0]]),
    "r": np.array([[7.0, 6.0], [5.0, 4.0], [3.0, 2.0]]),
}


@pytest.fixture()
def field_source(test_source):
    FIELD_SPECS = [
        {"param": param, "values": values.copy(), **MOCK_FIELD_METADATA} for param, values in DATA_VALUES.items()
    ]
    return test_source(FIELD_SPECS)


@pytest.fixture()
def ekd_from_source():
    def side_effect(source_type, path):
        mock_field = mock.Mock()
        if source_type != "file":
            raise ValueError("Invalid source type")
        # mask expected to be flattened
        mask = MASK_VALUES[path].copy().flatten()
        mock_field.to_numpy.return_value = mask
        return [mock_field]

    with mock.patch("anemoi.transform.filters.fields.apply_mask.ekd.from_source", autospec=True) as mock_fn:
        mock_fn.side_effect = side_effect
        yield mock_fn


@pytest.mark.parametrize(
    "threshold_options",
    [
        {"mask_value": 0.5},
        {"mask_value": 1},
        {"threshold": 0.5, "threshold_operator": ">"},
        {"threshold": 0.5, "threshold_operator": "<"},
    ],
)
@pytest.mark.parametrize("rename", [None, "renamed"])
@pytest.mark.parametrize("mask_name", MASK_VALUES.keys())
def test_apply_mask_fields(field_source, ekd_from_source, mask_name, rename, threshold_options):
    apply_mask = create_filter("apply_mask", path=mask_name, rename=rename, **threshold_options)
    ekd_from_source.assert_called_once_with("file", mask_name)

    pipeline = field_source | apply_mask

    input_fields = collect_fields_by_param(field_source)
    output_fields = collect_fields_by_param(pipeline)

    expected_mask = MASK_VALUES[mask_name].copy().flatten()
    if "mask_value" in threshold_options:
        # mask checks for exact equality - beware floats
        expected_mask = expected_mask == threshold_options["mask_value"]
    else:
        operator = {"<": np.less, ">": np.greater}[threshold_options["threshold_operator"]]
        expected_mask = operator(expected_mask, threshold_options["threshold"])
    expected_mask_count = np.sum(expected_mask)

    for param in DATA_VALUES.keys():
        result_param = f"{param}_{rename}" if rename else param
        assert result_param in output_fields
        for input_field, output_field in zip(input_fields[param], output_fields[result_param]):
            expected_values = input_field.to_numpy(flatten=True).copy()
            expected_values[expected_mask] = np.nan
            result = output_field.to_numpy(flatten=True)
            assert np.array_equal(expected_values, result, equal_nan=True)
            assert np.sum(np.isnan(result)) == expected_mask_count


def test_mask_tabular():
    config = {
        "col1": "lambda x: x >= 2",
    }
    df = pd.DataFrame({"col1": [0, 1, 2, 3], "col2": [3, 4, 5, 6]})
    mask = create_filter("mask", **config)
    result = mask(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["col2"].equals(df["col2"])
    expected_result = pd.Series([0, 1, np.nan, np.nan], name="col1")
    assert result["col1"].equals(expected_result)
