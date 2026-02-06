# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from tests.utils import create_tabular_filter as create_filter


def test_superob():
    config = {
        "grid": "o96",
        "timeslot_length": 3600,
        "columns_to_take_nearest": ["datetime"],
        "columns_to_groupby": ["reportype"],
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01 00:00:00"),
                pd.Timestamp("2025-01-01 00:00:01"),
                pd.Timestamp("2025-01-01 02:00:01"),
                pd.Timestamp("2025-01-01 02:00:02"),
            ],
            "latitude": [89.1, 89.3, 89.2, 89.2],
            "longitude": [-126, -126, -90, -90],
            "reportype": [1001, 1001, 1001, 1001],
            "obsvalue_rawbt_1": [207, 209, 265, 266],
        }
    )
    superob = create_filter("superob", **config)
    result = superob(df.copy())
    expect = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01 00:00:01"),
                pd.Timestamp("2025-01-01 02:00:01"),
            ],
            "latitude": [89.2, 89.2],
            "longitude": [-126.0, -90.0],
            "spatial_index": [13.0, 15.0],
            "reportype": [1001, 1001],
            "obsvalue_rawbt_1": [208.0, 265.5],
        }
    )
    print(result)
    print(expect)
    pd.testing.assert_frame_equal(
        result[expect.columns].reset_index(drop=True),
        expect.reset_index(drop=True),
        check_dtype=True,
        check_column_type=True,
        check_names=True,
    )


def test_superob_groupby():
    config = {
        "grid": "o96",
        "timeslot_length": 3600,
        "columns_to_take_nearest": ["datetime"],
        "columns_to_groupby": ["reportype"],
    }
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01 00:00:00"),
                pd.Timestamp("2025-01-01 00:00:01"),
                pd.Timestamp("2025-01-01 02:00:01"),
                pd.Timestamp("2025-01-01 02:00:02"),
            ],
            "latitude": [89.1, 89.3, 89.2, 89.2],
            "longitude": [-126.1, -126.3, -90, -90],
            "reportype": [1001, 1001, 1001, 1002],
            "obsvalue_rawbt_1": [207, 209, 265, 266],
        }
    )
    superob = create_filter("superob", **config)

    result = superob(df.copy())
    expect = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2025-01-01 00:00:00"),
                pd.Timestamp("2025-01-01 02:00:01"),
                pd.Timestamp("2025-01-01 02:00:02"),
            ],
            "latitude": [89.2, 89.2, 89.2],
            "longitude": [-126.2, -90.0, -90.0],
            "spatial_index": [13.0, 15.0, 15.0],
            "reportype": [1001, 1001, 1002],
            "obsvalue_rawbt_1": [208.0, 265.0, 266.0],
        }
    )
    print(result)
    print(expect)
    pd.testing.assert_frame_equal(
        result[expect.columns].reset_index(drop=True),
        expect.reset_index(drop=True),
        check_dtype=True,
        check_column_type=True,
        check_names=True,
    )
