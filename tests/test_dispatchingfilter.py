# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import pandas as pd
import pytest

from anemoi.transform.filter import DispatchingFilter

TEST_CASES = [
    pytest.param(pd.DataFrame(), id="dataframe"),
    pytest.param(ekd.FieldList(), id="fieldlist"),
    pytest.param(None, id="other"),
]


class ForwardField:
    def forward_fields(self, fields: ekd.FieldList) -> ekd.FieldList:
        return "fieldlist"


class ForwardTabular:
    def forward_tabular(self, df: pd.DataFrame) -> pd.DataFrame:
        return "dataframe"


class BackwardField:
    def backward_fields(self, fields: ekd.FieldList) -> ekd.FieldList:
        return "fieldlist reversed"


class BackwardTabular:
    def backward_tabular(self, df: pd.DataFrame) -> pd.DataFrame:
        return "dataframe reversed"


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_forward_fields_only(data):
    class ForwardFieldsOnly(ForwardField, DispatchingFilter):
        pass

    filter = ForwardFieldsOnly()
    match data:
        case ekd.FieldList():
            assert filter(data) == "fieldlist"
        case pd.DataFrame() | None:
            with pytest.raises(TypeError):
                filter(data)


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_forward_tabular_only(data):
    class ForwardTabularOnly(ForwardTabular, DispatchingFilter):
        pass

    filter = ForwardTabularOnly()
    match data:
        case pd.DataFrame():
            assert filter(data) == "dataframe"
        case ekd.FieldList() | None:
            with pytest.raises(TypeError):
                filter(data)


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_forward_both(data):
    class ForwardBoth(ForwardField, ForwardTabular, DispatchingFilter):
        pass

    filter = ForwardBoth()
    match data:
        case ekd.FieldList():
            assert filter(data) == "fieldlist"
        case pd.DataFrame():
            assert filter(data) == "dataframe"
        case None:
            with pytest.raises(TypeError):
                filter(data)


def test_dispatchingfilter_neither():
    with pytest.raises(TypeError):

        class Neither(DispatchingFilter):
            pass


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_backward_fields_only(data):
    class BackwardFieldsOnly(ForwardField, BackwardField, DispatchingFilter):
        pass

    filter = BackwardFieldsOnly().reverse()
    match data:
        case ekd.FieldList():
            assert filter(data) == "fieldlist reversed"
        case pd.DataFrame() | None:
            with pytest.raises(NotImplementedError):
                filter(data)


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_backward_tabular_only(data):
    class BackwardTabularOnly(ForwardTabular, BackwardTabular, DispatchingFilter):
        pass

    filter = BackwardTabularOnly().reverse()
    match data:
        case pd.DataFrame():
            assert filter(data) == "dataframe reversed"
        case ekd.FieldList() | None:
            with pytest.raises(NotImplementedError):
                filter(data)


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_backward_both(data):
    class BackwardBoth(ForwardField, ForwardTabular, BackwardField, BackwardTabular, DispatchingFilter):
        pass

    filter = BackwardBoth().reverse()
    match data:
        case ekd.FieldList():
            assert filter(data) == "fieldlist reversed"
        case pd.DataFrame():
            assert filter(data) == "dataframe reversed"
        case None:
            with pytest.raises(NotImplementedError):
                filter(data)


@pytest.mark.parametrize("data", TEST_CASES)
def test_dispatchingfilter_backward_neither(data):
    class BackwardNeither(ForwardField, ForwardTabular, DispatchingFilter):
        pass

    filter = BackwardNeither().reverse()
    with pytest.raises(NotImplementedError):
        filter(data)


def test_dispatchingfilter_mismatched_transforms():
    with pytest.raises(TypeError):

        class MismatchedTransforms1(ForwardField, BackwardTabular, DispatchingFilter):
            pass

    with pytest.raises(TypeError):

        class MismatchedTransform2(ForwardTabular, BackwardField, DispatchingFilter):
            pass
