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

from .utils import mock_field


@pytest.fixture
def sample_field():
    return ekd.from_source("sample", "test.grib")[0]


@pytest.mark.xfail(reason="setting arbitrary metadata not yet supported")
def test_field_new_metadata(sample_field):
    """Test that a new field can be created with new metadata."""
    # TODO: consider whether new_field_with_metadata should allow setting ekd field labels
    assert "foo" not in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert new_field.metadata("foo") == "bar"


def test_field_update_metadata(sample_field):
    """Test that a new field can be created with updated metadata."""
    assert sample_field.parameter.variable() == "2t"
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert new_field.parameter.variable() == "foo"


@pytest.mark.xfail(reason="centre key not currently supported")
def test_update_multiple_metadata(sample_field):
    """Test that we can update multiple metadata keys at once."""
    assert sample_field.metadata("param", "centre") == ("2t", "ecmf")
    new_field = new_field_with_metadata(sample_field, param="foo", centre="bar")
    assert new_field.metadata("param", "centre") == ("foo", "bar")


@pytest.mark.xfail(reason="setting arbitrary metadata not yet supported")
def test_metadata_in_new_field(sample_field):
    """Test that we can check if a key is in the metadata."""
    assert "foo" not in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert "foo" in new_field.metadata()


@pytest.mark.xfail(reason="unclear which metadata keys to fetch")
def test_field_with_updated_metadata_has_same_keys(sample_field):
    """Test that updating existing metadata leaves the keys unchanged."""
    assert "param" in sample_field.metadata()
    new_field = new_field_with_metadata(sample_field, param="foo")
    assert tuple(new_field.metadata().keys()) == tuple(sample_field.metadata().keys())


@pytest.mark.xfail(reason="unclear which metadata keys to fetch")
def test_field_adding_metadata_updates_keys(sample_field):
    """Test that adding a new metadata key is reflected in the keys."""
    assert "foo" not in tuple(sample_field.metadata().keys())
    new_field = new_field_with_metadata(sample_field, foo="bar")
    assert "foo" in tuple(new_field.metadata().keys())


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
    selection = FieldSelection(**{"parameter.variable": "2t", "vertical.level": 850})
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
