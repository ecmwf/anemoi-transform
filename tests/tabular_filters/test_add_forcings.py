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


def test_add_forcings():
    config = {
        "columns": [
            "cos_julian_day",
            "sin_julian_day",
            "cos_local_time",
            "sin_local_time",
            "cos_sza",
            "cos_latitude",
            "sin_latitude",
            "cos_longitude",
            "sin_longitude",
        ]
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01T00:00"),
                pd.Timestamp("2025-04-01T06:00"),
            ],
            "latitude": [-90.0, 90.0],
            "longitude": [0.0, 180.0],
        }
    )

    expected = {
        "cos_julian_day": [1.0, 0.018277],
        "sin_julian_day": [0.0, 1.0],
        "cos_local_time": [1.0, 0.0],
        "sin_local_time": [0.0, -1.0],
        "cos_sza": [0.391673, 0.075240],
        "cos_latitude": [0.0, 0.0],
        "sin_latitude": [-1.0, 1.0],
        "cos_longitude": [1.0, -1.0],
        "sin_longitude": [0.0, 0.0],
    }

    add_forcings = create_filter("add_forcings", **config)
    result = add_forcings(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(df.columns) | set(config["columns"])
    assert result.shape == (len(df), len(df.columns) + len(config["columns"]))

    assert result[list(df.columns)].equals(result[list(df.columns)])

    for col, expected_values in expected.items():
        assert np.allclose(result[col].to_numpy(), expected_values, rtol=1e-3)


def test_add_forcings_unknown_column():
    config = {
        "columns": [
            "cos_julian_day",
            "bad_column_name",
        ]
    }
    with pytest.raises(ValueError):
        _ = create_filter("add_forcings", **config)
