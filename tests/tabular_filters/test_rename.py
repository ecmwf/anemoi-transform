# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter


def test_rename():
    config = {
        "columns": {
            "x": "foo",
        }
    }
    df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [3, 4, 5],
        }
    )
    rename = create_filter("rename", **config)
    result = rename(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == ("foo", "y")
    assert result.shape == df.shape

    assert result["foo"].equals(df["x"])
    assert result["y"].equals(df["y"])


def test_rename_missing_column():
    config = {
        "columns": {
            "x": "foo",
        }
    }
    df = pd.DataFrame(
        {
            # x is missing
            "y": [3, 4, 5],
        }
    )
    rename = create_filter("rename", **config)
    with pytest.raises(ValueError):
        _ = rename(df.copy())
