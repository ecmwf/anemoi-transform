# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import numpy.testing as npt
import pytest

from anemoi.transform.filters import filter_registry

from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

# mask_param values: zeros at (0,0), (1,1), (2,1)
MASK_VALUES = np.array(
    [
        [0.0, 1.0],
        [2.0, 0.0],
        [1.0, 0.0],
    ]
)

PARAM_VALUES = np.array(
    [
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ]
)


@pytest.fixture()
def source(test_source):
    """Create a source with one mask_param field and one param field."""
    SPEC = [
        {"param": "snowc", "values": MASK_VALUES.copy(), **MOCK_FIELD_METADATA},
        {"param": "2t", "values": PARAM_VALUES.copy(), **MOCK_FIELD_METADATA},
    ]
    return test_source(SPEC)


def get_output_values(source, **filter_kwargs):
    """Apply the cross_mask filter and return the clipped param values."""
    f = filter_registry.create(
        "cross_mask",
        mask_param="snowc",
        param="2t",
        **filter_kwargs,
    )
    pipeline = source | f
    fields = collect_fields_by_param(pipeline)
    assert "2t" in fields, f"Expected '2t' in output, got: {set(fields)}"
    assert len(fields["2t"]) == 1
    return fields["2t"][0].to_numpy()


def test_eq_operator_clips_where_mask_close_to_comparison_value(source):
    """Where mask ≈ comparison_value (within atol), param is replaced with masked_value."""
    result = get_output_values(source, operator="eq", comparison_value=1.0, atol=0.0, masked_value=-999.0)

    # np.isclose(mask, comparison_value=1.0, atol=atol=0.0): True only for exact ones
    # Positions where MASK_VALUES == 1.0: (0,1), (2,0)=1.0
    expected = PARAM_VALUES.copy()
    expected[np.isclose(MASK_VALUES, 1.0, atol=0.0)] = -999.0

    npt.assert_allclose(result, expected)


def test_gt_operator_clips_where_mask_greater_than_comparison_value(source):
    """Where mask > comparison_value, param is replaced with masked_value."""
    result = get_output_values(source, operator="gt", comparison_value=1.0, masked_value=0.0)

    # Positions where MASK_VALUES > 1.0: (1,0)=2.0
    expected = PARAM_VALUES.copy()
    expected[MASK_VALUES > 1.0] = 0.0

    npt.assert_allclose(result, expected)


def test_lt_operator_clips_where_mask_less_than_comparison_value(source):
    """Where mask < comparison_value, param is replaced with masked_value."""
    result = get_output_values(source, operator="lt", comparison_value=1.0, masked_value=0.0)

    # Positions where MASK_VALUES < 1.0: (0,0)=0.0, (1,1)=0.0, (2,1)=0.0
    expected = PARAM_VALUES.copy()
    expected[MASK_VALUES < 1.0] = 0.0

    npt.assert_allclose(result, expected)


def test_ge_operator_clips_where_mask_greater_than_or_equal_to_comparison_value(source):
    """Where mask >= comparison_value, param is replaced with masked_value."""
    result = get_output_values(source, operator="ge", comparison_value=1.0, masked_value=0.0)

    # Positions where MASK_VALUES >= 1.0: (0,1)=1.0, (1,0)=2.0, (2,0)=1.0
    expected = PARAM_VALUES.copy()
    expected[MASK_VALUES >= 1.0] = 0.0

    npt.assert_allclose(result, expected)


def test_le_operator_clips_where_mask_less_than_or_equal_to_comparison_value(source):
    """Where mask <= comparison_value, param is replaced with masked_value."""
    result = get_output_values(source, operator="le", comparison_value=1.0, masked_value=0.0)

    # Positions where MASK_VALUES <= 1.0: (0,0), (0,1), (1,1), (2,0), (2,1)
    expected = PARAM_VALUES.copy()
    expected[MASK_VALUES <= 1.0] = 0.0

    npt.assert_allclose(result, expected)


def test_no_clipping_when_condition_never_met(source):
    """When no mask values satisfy the condition, param is unchanged."""
    # comparison_value=999.0: nothing in mask is > 999 → no clipping
    result = get_output_values(source, operator="gt", comparison_value=999.0, masked_value=0.0)

    npt.assert_allclose(result, PARAM_VALUES)


def test_all_values_clipped_when_condition_always_met(source):
    """When all mask values satisfy the condition, all param values are replaced."""
    # comparison_value=-999.0: everything in mask is > -999 → all clipped
    result = get_output_values(source, operator="gt", comparison_value=-999.0, masked_value=0.0)

    npt.assert_allclose(result, np.zeros_like(PARAM_VALUES))


def test_invalid_operator_raises_value_error(source):
    """An unsupported operator string raises a ValueError."""
    f = filter_registry.create(
        "cross_mask",
        mask_param="snowc",
        param="2t",
        comparison_value=0.0,
        masked_value=0.0,
        operator="invalid_op",
    )
    pipeline = source | f
    with pytest.raises(ValueError, match="Unsupported operator"):
        list(pipeline)


def test_mask_param_field_is_passed_through_unchanged(source):
    """The mask_param field is emitted unchanged by the filter."""
    f = filter_registry.create(
        "cross_mask",
        mask_param="snowc",
        param="2t",
        comparison_value=0.0,
        masked_value=0.0,
        operator="eq",
        atol=0.0,
    )
    pipeline = source | f
    fields = collect_fields_by_param(pipeline)

    assert "snowc" in fields
    npt.assert_allclose(fields["snowc"][0].to_numpy(), MASK_VALUES)


def test_ne_operator_clips_where_mask_not_close_to_comparison_value(source):
    """Where mask is NOT within atol of comparison_value, param is replaced."""
    result = get_output_values(source, operator="ne", comparison_value=0.0, atol=0.0, masked_value=0.0)

    # np.isclose(mask, comparison_value=0.0, atol=atol=0.0): True only for exact zeros
    # ne clips the complement: positions where mask != 0.0 → (0,1)=1.0, (1,0)=2.0, (2,0)=1.0
    expected = PARAM_VALUES.copy()
    expected[~np.isclose(MASK_VALUES, 0.0, atol=0.0)] = 0.0

    npt.assert_allclose(result, expected)
