import numpy as np
import pytest

from anemoi.transform.filter import SingleFieldFilter


@pytest.fixture
def source(test_source):
    return test_source("anemoi-transform/filters/cerra_20240601_single_level.grib")


def test_singlefieldfilter_cannot_be_instantiated():
    """Test that the SingleFieldFilter cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract method"):
        SingleFieldFilter()


def test_singlefieldfilter_invalid_required_inputs():
    """Test that the SingleFieldFilter raises an exception when required_inputs are an invalid type."""

    class TestFilter(SingleFieldFilter):
        required_inputs = "string_not_allowed"

        def forward_transform(self, field):
            pass

    with pytest.raises(TypeError, match="Required inputs must be a list or tuple"):
        TestFilter()


def test_singlefieldfilter_extra_inputs():
    """Test that the SingleFieldFilter raises an error when extra inputs are provided."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("foo",)

        def forward_transform(self, field):
            pass

    with pytest.raises(ValueError, match="Unknown input\(s\)"):
        TestFilter(foo="bar", baz="qux")


def test_singlefieldfilter_missing_required_inputs():
    """Test that the SingleFieldFilter raises an error when required inputs are missing."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("temperature",)

        def forward_transform(self, field):
            pass

    with pytest.raises(TypeError, match="Missing required input"):
        TestFilter()


def test_singlefieldfilter_defaults_for_optional_inputs():
    """Test that the SingleFieldFilter set optional inputs to their default values if not provided."""

    class TestFilter(SingleFieldFilter):
        optional_inputs = {"temperature": "2t"}

        def forward_transform(self, field):
            pass

    assert TestFilter().temperature == "2t"


def test_singlefieldfilter_defaults_are_overrideable():
    """Test that the SingleFieldFilter optional inputs can be overridden."""

    class TestFilter(SingleFieldFilter):
        optional_inputs = {"temperature": "2t"}

        def forward_transform(self, field):
            pass

    assert TestFilter(temperature="temperature").temperature == "temperature"


def test_singlefieldfilter_prepare():
    """Test that the SingleFieldFilter prepare method is called."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("positive_number",)

        def prepare_filter(self):
            if self.positive_number < 0:
                raise ValueError("positive_number must be positive")

        def forward_transform(self, field):
            pass

    with pytest.raises(ValueError, match="positive_number must be positive"):
        TestFilter(positive_number=-1)


def test_singlefieldfilter_only_forward_transform(source):
    """Test when the SingleFieldFilter only implements the forward_transform method."""

    class TestFilter(SingleFieldFilter):
        def forward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() + 1, template=field)

    pipeline = source | TestFilter()

    for original, result in zip(source, pipeline):
        assert np.allclose(original.to_numpy() + 1, result.to_numpy())


def test_singlefieldfilter_not_reversible_raises(source):
    """Test that the SingleFieldFilter raises an error when the filter is not reversible."""

    class TestFilter(SingleFieldFilter):
        def forward_transform(self, field):
            pass

    with pytest.raises(NotImplementedError, match="Field backward transform not implemented"):
        for _ in source | TestFilter.reversed():
            pass


def test_singlefieldfilter_forward_and_backward_transform(source):
    """Test that the SingleFieldFilter works transforms both forwards and backwards."""

    class TestFilter(SingleFieldFilter):
        def forward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() + 3, template=field)

        # deliberately not the inverse - expect difference of +1
        def backward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() - 2, template=field)

    filter = TestFilter()
    pipeline = source | filter | filter.reversed()

    for original, result in zip(source, pipeline):
        assert np.allclose(original.to_numpy() + 1, result.to_numpy())


def test_singlefieldfilter_simple_roundtrip(source):
    """Test that the SingleFieldFilter round trips successfully."""

    class TestFilter(SingleFieldFilter):
        def forward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() + 1, template=field)

        def backward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() - 1, template=field)

    filter = TestFilter()
    pipeline = source | filter | filter.reversed()

    for original, result in zip(source, pipeline):
        assert np.allclose(original.to_numpy(), result.to_numpy())


def test_singlefieldfilter_forward_select(source):
    """Test that the SingleFieldFilter is able to select on the forward transform."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("temperature",)

        def forward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() + 1, template=field)

        def forward_select(self):
            return {"param": self.temperature}

    # source dataset has 2t and 2r variables
    pipeline = source | TestFilter(temperature="2t")

    result_params = []
    for original, result in zip(source, pipeline):
        result_params.append(result.metadata("param"))
        assert original.metadata("param") == result.metadata("param")
        # only 2t has transform applied
        if result.metadata("param") == "2t":
            assert np.allclose(original.to_numpy() + 1, result.to_numpy())
        else:
            assert np.allclose(original.to_numpy(), result.to_numpy())
    assert set(result_params) == {"2t", "2r"}


def test_singlefieldfilter_backward_select(source):
    """Test that the SingleFieldFilter is able to select on the backward transform."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("temperature",)

        def forward_transform(self, field):
            pass

        def backward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() - 1, template=field)

        def backward_select(self):
            return {"param": self.temperature}

    # source dataset has 2t and 2r variables
    pipeline = source | TestFilter.reversed(temperature="2t")

    result_params = []
    for original, result in zip(source, pipeline):
        result_params.append(result.metadata("param"))
        assert original.metadata("param") == result.metadata("param")
        # only 2t has transform applied
        if result.metadata("param") == "2t":
            assert np.allclose(original.to_numpy() - 1, result.to_numpy())
        else:
            assert np.allclose(original.to_numpy(), result.to_numpy())
    assert set(result_params) == {"2t", "2r"}


def test_singlefieldfilter_backward_using_forward_select(source):
    """Test that the SingleFieldFilter is able to select in both directions with only forward select."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("temperature",)

        def forward_transform(self, field):
            pass

        def backward_transform(self, field):
            return self.new_field_from_numpy(field.to_numpy() - 1, template=field)

        def forward_select(self):
            return {"param": self.temperature}

    # source dataset has 2t and 2r variables
    pipeline = source | TestFilter.reversed(temperature="2t")

    result_params = []
    for original, result in zip(source, pipeline):
        result_params.append(result.metadata("param"))
        assert original.metadata("param") == result.metadata("param")
        if result.metadata("param") == "2t":
            assert np.allclose(original.to_numpy() - 1, result.to_numpy())
        else:
            assert np.allclose(original.to_numpy(), result.to_numpy())
    assert set(result_params) == {"2t", "2r"}


def test_singlefieldfilter_complex_roundtrip(source):
    """Test that the SingleFieldFilter is able to change both the data and metadata and select appropriately."""

    class TestFilter(SingleFieldFilter):
        required_inputs = ("temperature", "renamed_temperature")

        def forward_select(self):
            return {"param": self.temperature}

        def backward_select(self):
            return {"param": self.renamed_temperature}

        def forward_transform(self, field):
            new_metadata = {"param": self.renamed_temperature}
            return self.new_field_from_numpy(field.to_numpy() + 1, template=field, **new_metadata)

        def backward_transform(self, field):
            orig_metadata = {"param": self.temperature}
            return self.new_field_from_numpy(field.to_numpy() - 1, template=field, **orig_metadata)

    forward_filter = TestFilter(temperature="2t", renamed_temperature="2t_renamed")
    backward_filter = forward_filter.reversed(temperature="2t", renamed_temperature="2t_renamed")

    pipeline = source | forward_filter
    # forward transform
    for original, result in zip(source, pipeline):
        if original.metadata("param") == "2t":
            assert np.allclose(original.to_numpy() + 1, result.to_numpy())
            assert result.metadata("param") == "2t_renamed"
        else:
            assert np.allclose(original.to_numpy(), result.to_numpy())
            assert original.metadata("param") == result.metadata("param")

    # round trip
    pipeline = source | forward_filter | backward_filter
    for original, result in zip(source, pipeline):
        assert np.allclose(original.to_numpy(), result.to_numpy())
        assert original.metadata("param") == result.metadata("param")
