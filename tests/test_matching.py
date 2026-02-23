# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Iterator

import earthkit.data as ekd
import pytest

from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

from .utils import mock_field


class AddFields(MatchingFieldsFilter):
    @matching(select="param", forward=["a", "b"])
    def __init__(self, a, b, return_inputs="none"):
        self.a = a
        self.b = b
        self.return_inputs = return_inputs

    def forward_transform(self, a: ekd.Field, b: ekd.Field) -> Iterator[ekd.Field]:
        result = a.values + b.values
        yield self.new_field_from_numpy(result, template=a, param="c")

    def new_field_from_numpy(self, array, *, template, param):
        metadata = dict(template.metadata()) | {"param": param}
        return mock_field(**metadata)


def test_matching_decorator_initializes_correctly():
    filter_instance = AddFields("a", "b")
    assert filter_instance._initialised
    assert filter_instance.forward_arguments == {"a": "a", "b": "b"}


def test_forward_transform_adds_fields():
    a = mock_field(param="a", step=0, level=850)
    b = mock_field(param="b", step=0, level=850)
    data = ekd.SimpleFieldList([a, b])

    f = AddFields(a="a", b="b")
    result = f.forward(data)
    assert len(result) == 1
    assert isinstance(result[0], ekd.Field)
    assert result[0].metadata("param") == "c"


def test_return_inputs():
    a = mock_field(param="a", step=0, level=850)
    b = mock_field(param="b", step=0, level=850)
    data = ekd.SimpleFieldList([a, b])

    f = AddFields(a="a", b="b", return_inputs="all")
    result = f.forward(data)
    assert len(result) == 3
    for i in range(3):
        assert isinstance(result[i], ekd.Field)
    assert {result[i].metadata("param") for i in range(2)} == {"a", "b"}
    assert result[2].metadata("param") == "c"

    f = AddFields(a="a", b="b", return_inputs=["a"])
    result = f.forward(data)
    assert len(result) == 2
    for i in range(2):
        assert isinstance(result[i], ekd.Field)
    assert result[0].metadata("param") == "a"
    assert result[1].metadata("param") == "c"


def test_missing_component_raises():
    a = mock_field(param="a", step=0, level=850)
    # Missing 'b'
    data = ekd.SimpleFieldList([a])
    f = AddFields(a="a", b="b")

    with pytest.raises(ValueError):
        _ = f.forward(data)


def test_uninitialised_filter_raises():
    class BadFilter(MatchingFieldsFilter):
        def forward_transform(self, *args):
            pass

    bf = BadFilter()
    with pytest.raises(ValueError):
        _ = bf.forward_arguments


def test_metadata_mismatch_warning(caplog):
    c = mock_field(param="c", step=0, level=850)
    d = mock_field(param="d", step=0, level=850)
    data = ekd.SimpleFieldList([c, d])

    f = AddFields(a="a", b="b")

    with caplog.at_level("WARNING"):
        f.forward(data)

    assert "Please ensure your filter is configured to match the input variables metadata" in caplog.text
