# (C) Copyright 2026- Anemoi contributors.
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

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

MASK_VALUES = {
    "lsm_zeros": np.array([[0, 0], [0, 0], [0, 0]]),
    "lsm_ones": np.array([[1, 1], [1, 1], [1, 1]]),
    "lsm_mixed_ints": np.array([[0, 1], [1, 0], [1, 2]]),
    "lsm_mixed_floats": np.array([[0.0, 0.25], [0.5, 0.5], [0.75, 1.0]]),
}

DATA_VALUES = {
    "t": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    "q": np.array([[7.0, 8.0], [9.0, 0.0], [9.0, 8.0]]),
}


@pytest.fixture
def source(test_source):
    def _create_source(mask_name):
        FIELD_SPECS = [
            {"param": param, "values": values.copy(), **MOCK_FIELD_METADATA} for param, values in DATA_VALUES.items()
        ]
        FIELD_SPECS.append({"param": "lsm", "values": MASK_VALUES[mask_name].copy(), **MOCK_FIELD_METADATA})
        return test_source(FIELD_SPECS)

    return _create_source


def test_mask_with_field_fails_without_arguments():
    with pytest.raises(ValueError, match="Either `mask_value` or `threshold` must be provided."):
        create_filter("mask_with_field", param="t", mask_param="lsm")


@pytest.mark.parametrize(
    "threshold_options",
    [
        {"mask_value": 0},
        {"mask_value": 1},
        {"threshold": 0.5, "threshold_operator": ">"},
        {"threshold": 0.5, "threshold_operator": "<="},
    ],
)
@pytest.mark.parametrize("rename", [None, "renamed"])
@pytest.mark.parametrize("mask_name", MASK_VALUES.keys())
def test_mask_with_field_forward(source, mask_name, rename, threshold_options):
    src = source(mask_name)
    mask_with_field = create_filter("mask_with_field", param="t", mask_param="lsm", rename=rename, **threshold_options)

    pipeline = src | mask_with_field

    input_fields = collect_fields_by_param(src)
    output_fields = collect_fields_by_param(pipeline)

    expected_mask_values = MASK_VALUES[mask_name].copy().flatten()
    if "mask_value" in threshold_options:
        expected_mask = expected_mask_values == threshold_options["mask_value"]
    else:
        from anemoi.transform.filters.fields.mask_with_field import OPERATORS

        operator = OPERATORS[threshold_options["threshold_operator"]]
        expected_mask = operator(expected_mask_values, threshold_options["threshold"])

    expected_mask_count = np.sum(expected_mask)

    # Check that 't' is masked
    result_param = f"t_{rename}" if rename else "t"
    assert result_param in output_fields
    for input_field, output_field in zip(input_fields["t"], output_fields[result_param]):
        expected_values = input_field.to_numpy(flatten=True).copy()
        expected_values[expected_mask] = np.nan
        result = output_field.to_numpy(flatten=True)
        assert np.array_equal(expected_values, result, equal_nan=True)
        assert np.sum(np.isnan(result)) == expected_mask_count

    # Check that 'q' is untouched
    assert "q" in output_fields
    for input_field, output_field in zip(input_fields["q"], output_fields["q"]):
        assert np.array_equal(input_field.to_numpy(flatten=True), output_field.to_numpy(flatten=True))

    # Check that 'lsm' is returned
    assert "lsm" in output_fields
    for input_field, output_field in zip(input_fields["lsm"], output_fields["lsm"]):
        assert np.array_equal(input_field.to_numpy(flatten=True), output_field.to_numpy(flatten=True))


def test_mask_with_field_custom_mask_param(source):
    # Test using a different parameter as mask
    FIELD_SPECS = [
        {"param": "t", "values": np.array([1.0, 2.0]), "latitudes": [0, 0], "longitudes": [0, 1]},
        {"param": "my_mask", "values": np.array([0, 1]), "latitudes": [0, 0], "longitudes": [0, 1]},
    ]
    import earthkit.data as ekd

    from anemoi.transform.sources import source_registry

    ds = ekd.from_source("list-of-dicts", FIELD_SPECS)
    src = source_registry.create("testing", dataset=ds)

    filter = create_filter("mask_with_field", param="t", mask_param="my_mask", mask_value=0)
    pipeline = src | filter
    output_fields = collect_fields_by_param(pipeline)

    assert "t" in output_fields
    result = output_fields["t"][0].to_numpy(flatten=True)
    assert np.isnan(result[0])
    assert result[1] == 2.0


def test_mask_with_field_drop_mask(source):
    # Test drop_mask=True
    src = source("lsm_mixed_ints")
    mask_with_field = create_filter("mask_with_field", param="t", mask_param="lsm", mask_value=0, drop_mask=True)

    pipeline = src | mask_with_field
    output_fields = collect_fields_by_param(pipeline)

    # 't' should be there and masked
    assert "t" in output_fields
    # 'q' should still be there (unmasked)
    assert "q" in output_fields
    # 'lsm' should not be there
    assert "lsm" not in output_fields


def test_mask_with_field_drop_mask_with_rename(source):
    # Test drop_mask=True with rename
    src = source("lsm_mixed_ints")
    mask_with_field = create_filter(
        "mask_with_field", param="t", mask_param="lsm", mask_value=0, drop_mask=True, rename="masked"
    )

    pipeline = src | mask_with_field
    output_fields = collect_fields_by_param(pipeline)

    # 'lsm' should not be there
    assert "lsm" not in output_fields


def test_mask_with_field_no_drop_mask_default(source):
    # Test drop_mask=False (default)
    src = source("lsm_mixed_ints")
    mask_with_field = create_filter("mask_with_field", param="t", mask_param="lsm", mask_value=0)

    pipeline = src | mask_with_field
    output_fields = collect_fields_by_param(pipeline)

    # 't' should be there
    assert "t" in output_fields
    # 'lsm' should be there
    assert "lsm" in output_fields
