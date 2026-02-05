# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter


def test_sort_by():
    config = {
        "columns": [
            "col1",
            "col2",
        ]
    }
    df = pd.DataFrame(
        {
            "col1": np.array([2, 1, 2, 3, 3]),
            "col2": np.array([4, 5, 3, 2, 1]),
            "col3": np.array([0, 1, 2, 3, 4]),
        }
    )
    sort_by = create_filter("sort_by", **config)
    result = sort_by(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    expected = pd.DataFrame(
        {
            "col1": np.array([1, 2, 2, 3, 3]),
            "col2": np.array([5, 3, 4, 1, 2]),
            "col3": np.array([1, 2, 0, 4, 3]),
        }
    )
    # same aside from index which will have been reordered
    assert result.reset_index(drop=True).equals(expected)


def test_sort_by_missing_columns():
    config = {
        "columns": [
            "col1",
        ]
    }
    df = pd.DataFrame(
        {
            # col1 missing
            "col2": np.array([4, 5, 3, 2, 1]),
            "col3": np.array([0, 1, 2, 3, 4]),
        }
    )
    sort_by = create_filter("sort_by", **config)
    with pytest.raises(ValueError):
        _ = sort_by(df.copy())
