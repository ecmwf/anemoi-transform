# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Iterator

import numpy as np
import pytest

from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


class MockField:
    def __init__(self, param, **meta):
        self._param = param
        self._meta = meta
        self.values = np.array([1.0])  # dummy data

    def metadata(self, namespace=None):
        if namespace == "mars":
            return dict(self._meta, param=self._param)
        return self._param


class MockFieldList(list):
    def metadata(self, name):
        return [getattr(f, "metadata")("mars")[name] for f in self]


class AddFields(MatchingFieldsFilter):
    @matching(select="param", forward=["a", "b"])
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward_transform(self, a: MockField, b: MockField) -> Iterator[MockField]:
        result = a.values + b.values
        yield self.new_field_from_numpy(result, template=a, param="c")

    def new_field_from_numpy(self, array, *, template, param):
        return MockField(param, **template._meta)

    def new_fieldlist_from_list(self, fields):
        return MockFieldList(fields)


def test_matching_decorator_initializes_correctly():
    filter_instance = AddFields("a", "b")
    assert filter_instance._initialised
    assert filter_instance.forward_arguments == {"a": "a", "b": "b"}


def test_forward_transform_adds_fields():
    a = MockField("a", step=0, level=850)
    b = MockField("b", step=0, level=850)
    data = MockFieldList([a, b])

    f = AddFields(a="a", b="b")
    result = f.forward(data)
    assert len(result) == 1
    assert isinstance(result[0], MockField)
    assert result[0]._param == "c"


def test_missing_component_raises():
    a = MockField("a", step=0, level=850)
    # Missing 'b'
    data = MockFieldList([a])
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
