import pytest

from src.anemoi.transform.fields import FieldSelection


class MockField:
    def __init__(self, **metadata):
        self._metadata = metadata

    # FieldSelection only sends the metadata message to objects
    def metadata(self, key):
        return self._metadata[key]


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
