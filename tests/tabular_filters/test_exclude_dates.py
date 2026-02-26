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


def test_exclude_dates():
    config = {
        "col1": [
            [20250101, 20250102],
            [20250105, 20250105],
        ]
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01T00:00"),
                pd.Timestamp("2025-01-02T00:00"),
                pd.Timestamp("2025-01-02T06:00"),
                pd.Timestamp("2025-01-03T00:00"),
                pd.Timestamp("2025-05-04T00:00"),
            ],
            "col1": np.array([0, 1, 2, 3, 4]),
            "col2": np.array([0, 1, 2, 3, 4]),
        }
    )

    exclude_dates = create_filter("exclude_dates", **config)
    result = exclude_dates(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result[["datetime", "col2"]].equals(df[["datetime", "col2"]])

    expected = (np.array([np.nan, np.nan, np.nan, 3, 4]),)
    assert np.allclose(result["col1"].to_numpy(), expected, equal_nan=True)


def test_exclude_dates_single_range():
    config = {
        "col1": [20250101, 20250101],
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01T00:00"),
                pd.Timestamp("2025-01-02T00:00"),
                pd.Timestamp("2025-01-02T06:00"),
                pd.Timestamp("2025-01-03T00:00"),
                pd.Timestamp("2025-05-04T00:00"),
            ],
            "col1": np.array([0, 1, 2, 3, 4]),
            "col2": np.array([0, 1, 2, 3, 4]),
        }
    )

    exclude_dates = create_filter("exclude_dates", **config)
    result = exclude_dates(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result[["datetime", "col2"]].equals(df[["datetime", "col2"]])

    expected = (np.array([np.nan, 1, 2, 3, 4]),)
    assert np.allclose(result["col1"].to_numpy(), expected, equal_nan=True)


def test_exclude_dates_missing_column():
    config = {
        "col1": [
            [20250101, 20250102],
            [20250105, 20250105],
        ]
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01T00:00"),
                pd.Timestamp("2025-01-02T00:00"),
                pd.Timestamp("2025-01-02T06:00"),
                pd.Timestamp("2025-01-03T00:00"),
                pd.Timestamp("2025-05-04T00:00"),
            ],
            # col1 is missing
            "col2": np.array([0, 1, 2, 3, 4]),
        }
    )

    exclude_dates = create_filter("exclude_dates", **config)
    with pytest.raises(ValueError):
        _ = exclude_dates(df.copy())


def test_exclude_dates_no_config():
    config = {}
    with pytest.raises(ValueError):
        _ = create_filter("exclude_dates", **config)
