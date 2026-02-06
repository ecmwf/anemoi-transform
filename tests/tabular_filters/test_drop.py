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


def test_drop():
    config = {
        "columns": ["drop_me"],
    }
    df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "drop_me": [3, 4, 5],
        }
    )
    drop = create_filter("drop", **config)
    result = drop(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == ("x",)
    assert result.shape == (len(df), len(df.columns) - len(config["columns"]))

    assert result["x"].equals(df["x"])


def test_drop_no_columns():
    config = {
        "columns": [],
    }
    with pytest.raises(ValueError):
        _ = create_filter("drop", **config)


def test_drop_missing_column():
    config = {
        "columns": ["drop_me"],
    }
    df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            # drop_me is missing
        }
    )
    drop = create_filter("drop", **config)
    with pytest.raises(ValueError):
        _ = drop(df.copy())
