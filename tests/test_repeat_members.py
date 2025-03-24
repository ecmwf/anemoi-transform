# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from typing import Any
from typing import Tuple

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi.transform.filters.repeat_members import RepeatMembers

NO_MARS = not os.path.exists(os.path.expanduser("~/.ecmwfapirc"))


def _get_template() -> Tuple[Any, np.ndarray, Any]:
    """Get a template fieldlist, values, and metadata for testing.

    Returns
    -------
    Tuple
        A tuple containing the fieldlist, values, and metadata.
    """
    temp = ekd.from_source("mars", {"param": "2t", "levtype": "sfc", "dates": ["2023-11-17 00:00:00"]})
    fieldlist = temp.to_fieldlist()
    return fieldlist, fieldlist[0].values, fieldlist[0].metadata


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_repeat_members_using_numbers_1() -> None:
    """Test RepeatMembers filter using a list of numbers.

    Tests:
    - Repeating members using a list of numbers [1, 2, 3].
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(numbers=[1, 2, 3])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_repeat_members_using_numbers_2() -> None:
    """Test RepeatMembers filter using a range of numbers.

    Tests:
    - Repeating members using a range of numbers "1/to/3".
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(numbers="1/to/3")
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_repeat_members_using_members() -> None:
    """Test RepeatMembers filter using a list of members.

    Tests:
    - Repeating members using a list of members [0, 1, 2].
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(members=[0, 1, 2])
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_repeat_members_using_count() -> None:
    """Test RepeatMembers filter using a count.

    Tests:
    - Repeating members using a count of 3.
    - Asserting the repeated members have correct values and metadata.
    """
    fieldlist, values, metadata = _get_template()

    repeat = RepeatMembers(count=3)
    repeated = repeat.forward(fieldlist)
    assert len(repeated) == 3
    for i, f in enumerate(repeated):
        assert f.values.shape == values.shape
        assert np.all(f.values == values)
        assert f.metadata("number") == i + 1
        assert f.metadata("name") == metadata("name")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
