# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform import FieldList
from anemoi.transform.filters import create_filter_by_name as create_filter


@pytest.fixture
def template(fieldlist: FieldList) -> tuple[FieldList, np.ndarray]:
    """Fixture returning a single-field (2t) fieldlist and its values."""
    single = fieldlist.sel(**{"parameter.variable": "2t"})
    assert len(single) == 1
    return single, single[0].values


@skip_if_offline
def test_repeat_members_using_numbers_1(template: tuple[FieldList, np.ndarray]) -> None:
    """Test repeat_members filter using a list of numbers.

    Tests:
    - Repeating members using a list of numbers [1, 2, 3].
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values = template

    repeat = create_filter("repeat_members", numbers=[1, 2, 3])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.ensemble.member() == str(i + 1)


@skip_if_offline
def test_repeat_members_using_numbers_2(template: tuple[FieldList, np.ndarray]) -> None:
    """Test repeat_members filter using a range of numbers.

    Tests:
    - Repeating members using a range of numbers "1/to/3".
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values = template

    repeat = create_filter("repeat_members", numbers="1/to/3")
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.ensemble.member() == str(i + 1)


@skip_if_offline
def test_repeat_members_using_members(template: tuple[FieldList, np.ndarray]) -> None:
    """Test repeat_members filter using a list of members.

    Tests:
    - Repeating members using a list of members [0, 1, 2].
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values = template

    repeat = create_filter("repeat_members", members=[0, 1, 2])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.ensemble.member() == str(i + 1)


@skip_if_offline
def test_repeat_members_using_count(template: tuple[FieldList, np.ndarray]) -> None:
    """Test repeat_members filter using a count.

    Tests:
    - Repeating members using a count of 3.
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values = template

    repeat = create_filter("repeat_members", count=3)
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.ensemble.member() == str(i + 1)


if __name__ == "__main__":
    pytest.main([__file__])
