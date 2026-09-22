# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import pytest

from anemoi.transform.fields import FieldSelection
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.metadata import get_metadata

from .utils import mock_field


@pytest.fixture
def sample_field():
    return ekd.from_source("sample", "test.grib").to_fieldlist()[0]


def test_field_new_metadata(sample_field):
    """Test that a new field can be created with new metadata."""
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert new_field.get("labels.foo") == "bar"


def test_field_update_metadata(sample_field):
    """Test that a new field can be created with updated metadata."""
    assert sample_field.parameter.variable() == "2t"
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert new_field.parameter.variable() == "foo"


@pytest.mark.xfail(reason="raw GRIB keys must be set explicitly as 'metadata.centre'")
def test_update_multiple_metadata(sample_field):
    """Test that we can update multiple metadata keys at once."""
    assert sample_field.metadata("param") == "2t"
    assert sample_field.metadata("centre") == "ecmf"
    new_field = new_field_with_metadata(sample_field, param="foo", centre="bar")
    assert new_field.parameter.variable() == "foo"
    assert new_field.metadata("centre") == "bar"


def test_metadata_in_new_field(sample_field):
    """Test that we can check if a key is in the metadata."""
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert new_field.get("labels.foo") == "bar"


@pytest.mark.xfail(reason="no dict-like metadata interface")
def test_field_with_updated_metadata_has_same_keys(sample_field):
    """Test that updating existing metadata leaves the keys unchanged."""
    assert sample_field.metadata("param") == "2t"
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert new_field.parameter.variable() == "foo"
    assert new_field.metadata("param") == "foo"


def test_field_adding_metadata_updates_keys(sample_field):
    """Test that adding a new metadata key is reflected in the keys."""
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert new_field.get("labels.foo") == "bar"


def test_field_arbitrary_metadata_stored_as_label(sample_field):
    """Arbitrary keys are stored under the ``labels`` namespace and read back."""
    new_field = new_field_with_metadata(sample_field, anemoi_origin="mars")
    assert new_field.get("labels.anemoi_origin") == "mars"
    assert new_field.labels == {"anemoi_origin": "mars"}
    assert get_metadata(new_field, "anemoi_origin") == "mars"


def test_get_metadata_missing_key(sample_field):
    """An unresolvable key raises KeyError, or returns the default if given."""
    with pytest.raises(KeyError):
        get_metadata(sample_field, "no_such_key")

    assert get_metadata(sample_field, "no_such_key", default=None) is None
    assert get_metadata(sample_field, "no_such_key", default="fallback") == "fallback"


def test_field_arbitrary_metadata_accumulates(sample_field):
    """Successive calls add labels rather than replacing the whole namespace."""
    new_field = new_field_with_metadata(sample_field, anemoi_origin="mars")
    new_field = new_field_with_metadata(new_field, anemoi_extra="other")
    assert new_field.labels == {"anemoi_origin": "mars", "anemoi_extra": "other"}


def test_field_component_key_passed_through(sample_field):
    """A component key is used as-is rather than being treated as a label."""
    new_field = new_field_with_metadata(sample_field, **{"parameter.variable": "foo"})
    assert new_field.parameter.variable() == "foo"
    assert not new_field.labels


def test_field_mixed_metadata_keys(sample_field):
    """MARS, component and arbitrary keys can be mixed in a single call."""
    new_field = new_field_with_metadata(
        sample_field,
        param="foo",
        anemoi_origin="mars",
        **{"vertical.level": 500},
    )
    assert new_field.parameter.variable() == "foo"
    assert new_field.vertical.level() == 500
    assert new_field.labels == {"anemoi_origin": "mars"}


def test_fieldselection_match_all():
    """Test FieldSelection with no arguments matches all fields."""
    field = mock_field(**{"labels.invalid_key": "value"})
    selection = FieldSelection()
    assert selection.match(field)


def test_fieldselection_invalid_key():
    """Test FieldSelection raises an exception with an invalid key."""
    with pytest.raises(ValueError, match="Invalid keys in spec"):
        FieldSelection(invalid_key="value")


def test_fieldselection_match_fail_different_param():
    """Test FieldSelection match fails when param is different."""
    field = mock_field(**{"parameter.variable": "2t"})
    selection = FieldSelection(**{"parameter.variable": "2z"})
    assert not selection.match(field)


def test_fieldselection_match_same_param():
    """Test FieldSelection match succeeds when param is the same."""
    field = mock_field(**{"parameter.variable": "2t"})
    selection = FieldSelection(**{"parameter.variable": "2t"})
    assert selection.match(field)


def test_fieldselection_match_fail_missing_key():
    """Test FieldSelection match fails when a selection key is missing on the field."""
    field = mock_field(**{"parameter.variable": "t"})
    selection = FieldSelection(**{"parameter.variable": "t", "vertical.level": 850})
    assert not selection.match(field)


def test_fieldselection_match_field_with_extra_metadata():
    """Test FieldSelection match succeeds when the field has extra metadata."""
    field = mock_field(**{"parameter.variable": "t", "vertical.level": 850})
    selection = FieldSelection(**{"parameter.variable": "t"})
    assert selection.match(field)


def test_fieldselection_match_fail_same_param_different_level():
    """Test FieldSelection match fails when param is the same but the levelist is different."""
    field = mock_field(**{"parameter.variable": "t", "vertical.level": 100})
    selection = FieldSelection(**{"parameter.variable": "t", "vertical.level": 850})
    assert not selection.match(field)


def test_fieldselection_match_same_param_same_level():
    """Test FieldSelection match succeeds when param and level are the same."""
    field = mock_field(**{"parameter.variable": "t", "vertical.level": 850})
    selection = FieldSelection(**{"parameter.variable": "t", "vertical.level": 850})
    assert selection.match(field)


def test_fieldselection_match_is_subset():
    """Test FieldSelection match succeeds when the field is a subset of the selection."""
    field = mock_field(**{"parameter.variable": "t", "vertical.level": 850})
    selection = FieldSelection(**{"parameter.variable": ["t", "q"], "vertical.level": [850, 950]})
    assert selection.match(field)


def test_fieldselection_match_fail_different_param_same_level():
    """Test FieldSelection match fails when the is on the same level but a different param."""
    field = mock_field(**{"parameter.variable": "t", "vertical.level": 850})
    selection = FieldSelection(**{"parameter.variable": "q", "vertical.level": [850, 950]})
    assert not selection.match(field)
