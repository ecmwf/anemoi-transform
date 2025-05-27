# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import pytest

from anemoi.transform.fields import new_field_with_metadata
from src.anemoi.transform.fields import FieldSelection


class MockField:
    def __init__(self, **metadata):
        self._metadata = metadata

    # FieldSelection only sends the metadata message to objects
    def metadata(self, key):
        return self._metadata[key]


@pytest.fixture
def sample_field():
    return ekd.from_source("sample", "test.grib")[0]


def test_field_new_metadata(sample_field):
    """Test that a new field can be created with new metadata."""
    assert "foo" not in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert new_field.metadata("foo") == "bar"


def test_field_update_metadata(sample_field):
    """Test that a new field can be created with updated metadata."""
    assert sample_field.metadata("param") == "2t"
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert new_field.metadata("param") == "foo"


def test_update_multiple_metadata(sample_field):
    """Test that we can update multiple metadata keys at once."""
    assert sample_field.metadata("param", "centre") == ("2t", "ecmf")
    new_field = new_field_with_metadata(sample_field, param="foo", centre="bar")
    assert new_field.metadata("param", "centre") == ("foo", "bar")


@pytest.mark.xfail(reason="__contains__ not yet implemented for metadata")
def test_metadata_in_new_field(sample_field):
    """Test that we can check if a key is in the metadata."""
    assert "foo" not in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert "foo" in new_field.metadata()


def test_field_with_updated_metadata_has_same_keys(sample_field):
    """Test that updating existing metadata leaves the keys unchanged."""
    assert "param" in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert tuple(new_field.metadata().keys()) == tuple(sample_field.metadata().keys())


@pytest.mark.xfail(reason="updating metadata keys not yet implemented")
def test_field_adding_metadata_updates_keys(sample_field):
    """Test that adding a new metadata key is reflected in the keys."""
    assert "foo" not in tuple(sample_field.metadata().keys())
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert "foo" in tuple(new_field.metadata().keys())


def test_fieldselection_match_all():
    """Test FieldSelection with no arguments matches all fields."""
    field = MockField(invalid_key="any_value")
    selection = FieldSelection()
    assert selection.match(field)


def test_fieldselection_invalid_key():
    """Test FieldSelection raises an exception with an invalid key."""
    with pytest.raises(ValueError, match="Invalid keys in spec"):
        FieldSelection(invalid_key="value")


def test_fieldselection_match_fail_different_param():
    """Test FieldSelection match fails when param is different."""
    field = MockField(param="2t")
    selection = FieldSelection(param="2z")
    assert not selection.match(field)


def test_fieldselection_match_same_param():
    """Test FieldSelection match succeeds when param is the same."""
    field = MockField(param="2t")
    selection = FieldSelection(param="2t")
    assert selection.match(field)


def test_fieldselection_match_fail_missing_key():
    """Test FieldSelection match fails when a selection key is missing on the field."""
    field = MockField(param="t")
    selection = FieldSelection(param="t", levelist=850)
    assert not selection.match(field)


def test_fieldselection_match_field_with_extra_metadata():
    """Test FieldSelection match succeeds when the field has extra metadata."""
    field = MockField(param="t", levelist=850)
    selection = FieldSelection(param="t")
    assert selection.match(field)


def test_fieldselection_match_fail_same_param_different_level():
    """Test FieldSelection match fails when param is the same but the levelist is different."""
    field = MockField(param="t", levelist=100)
    selection = FieldSelection(param="t", levelist=850)
    assert not selection.match(field)


def test_fieldselection_match_same_param_same_level():
    """Test FieldSelection match succeeds when param and level are the same."""
    field = MockField(param="t", levelist=850)
    selection = FieldSelection(param="t", levelist=850)
    assert selection.match(field)


def test_fieldselection_match_is_subset():
    """Test FieldSelection match succeeds when the field is a subset of the selection."""
    field = MockField(param="t", levelist=850)
    selection = FieldSelection(param=["t", "q"], levelist=[850, 950])
    assert selection.match(field)


def test_fieldselection_match_fail_different_param_same_level():
    """Test FieldSelection match fails when the is on the same level but a different param."""
    field = MockField(param="t", levelist=850)
    selection = FieldSelection(param="q", levelist=[850, 950])
    assert not selection.match(field)
