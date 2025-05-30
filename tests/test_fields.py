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
