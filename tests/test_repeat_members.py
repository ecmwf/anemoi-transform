# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters.repeat_members import RepeatMembers


def _get_template():
    temp = ekd.from_source("mars", {"param": "2t", "levtype": "sfc", "dates": ["2023-11-17 00:00:00"]})
    fieldlist = temp.to_fieldlist()
    return fieldlist, fieldlist[0].values, fieldlist[0].metadata


def test_repeat_members_using_numbers_1():
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(numbers=[1, 2, 3])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


def test_repeat_members_using_numbers_2():
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(numbers="1/to/3")
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


def test_repeat_members_using_members():
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(members=[0, 1, 2])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


def test_repeat_members_using_count():
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(count=3)
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")
