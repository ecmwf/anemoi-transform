# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from unittest.mock import patch

import earthkit.data as ekd
import numpy as np
import pandas as pd
import pytest

from anemoi.transform.filters import create_filter_by_name as create_filter
from anemoi.transform.filters.tabular.irregular_to_grid import IrregularToGrid


@pytest.fixture
def small_grid():
    lats = np.array([45.0, 45.0, -45.0, -45.0])
    lons = np.array([0.0, 90.0, 0.0, 90.0])
    return lats, lons


@pytest.fixture
def mock_define_grid(small_grid):
    with patch.object(IrregularToGrid, "_define_grid", return_value=small_grid):
        yield


@pytest.fixture
def mock_ekd_template(small_grid):
    lats, lons = small_grid
    template_fl = ekd.from_source(
        "list-of-dicts",
        [
            {
                "param": "t",
                "values": np.zeros(len(lats)),
                "latitudes": lats.tolist(),
                "longitudes": lons.tolist(),
                "valid_datetime": "2023-01-01T00:00:00Z",
            }
        ],
    )
    with patch("earthkit.data.from_source", return_value=template_fl) as mock_from_source:
        yield mock_from_source


@pytest.mark.parametrize(
    "config, df, expected_arrays",
    [
        pytest.param(
            {
                "template": "dummy.grib",
                "start_time": datetime(2023, 1, 1, 0, 0, 0),
                "end_time": datetime(2023, 1, 1, 0, 0, 0),
                "time_freq": "6h",
                "columns": ["temperature", "humidity"],
            },
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2023-01-01 05:00",
                            "2023-01-01 05:30",
                            "2023-01-01 05:00",
                            "2023-01-01 11:00",
                            "2023-01-01 11:30",
                        ]
                    ),
                    "spatial_index": [0, 1, 2, 0, 3],
                    "temperature": [10.0, 20.0, 30.0, 15.0, 25.0],
                    "humidity": [50.0, 60.0, 70.0, 55.0, 65.0],
                }
            ),
            {
                "temperature": np.array(
                    [
                        [10.0, 20.0, 30.0, np.nan],
                        [15.0, np.nan, np.nan, 25.0],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                "humidity": np.array(
                    [
                        [50.0, 60.0, 70.0, np.nan],
                        [55.0, np.nan, np.nan, 65.0],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            },
            id="values_mapped_to_correct_time_and_spatial_positions",
        ),
        pytest.param(
            # window for target 06:00 is (00:00 exclusive, 06:01 inclusive]; all three obs fall in it
            # nearest to 06:00 is 05:50 (10 min away), so 200.0 is selected over 300.0 (60 min) and 100.0 (180 min)
            {
                "template": "dummy.grib",
                "start_time": datetime(2023, 1, 1, 0, 0, 0),
                "end_time": datetime(2023, 1, 1, 0, 0, 0),
                "time_freq": "6h",
                "columns": ["temperature"],
            },
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2023-01-01 03:00",
                            "2023-01-01 05:50",
                            "2023-01-01 05:00",
                        ]
                    ),
                    "spatial_index": [0, 0, 0],
                    "temperature": [100.0, 200.0, 300.0],
                }
            ),
            {
                "temperature": np.array(
                    [
                        [200.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            },
            id="nearest_observation_selected_in_window",
        ),
        pytest.param(
            # rows where ALL columns are NaN are dropped entirely; rows where only SOME columns are NaN
            # are kept – the NaN value is written to the grid for that column
            {
                "template": "dummy.grib",
                "start_time": datetime(2023, 1, 1, 0, 0, 0),
                "end_time": datetime(2023, 1, 1, 0, 0, 0),
                "time_freq": "6h",
                "columns": ["temperature", "humidity"],
            },
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2023-01-01 05:00",  # spatial_index=0: all NaN → excluded
                            "2023-01-01 05:30",  # spatial_index=1: temperature valid, humidity NaN → kept
                            "2023-01-01 05:00",  # spatial_index=2: temperature NaN, humidity valid → kept
                        ]
                    ),
                    "spatial_index": [0, 1, 2],
                    "temperature": [np.nan, 20.0, np.nan],
                    "humidity": [np.nan, np.nan, 70.0],
                }
            ),
            {
                "temperature": np.array(
                    [
                        [np.nan, 20.0, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
                "humidity": np.array(
                    [
                        [np.nan, np.nan, 70.0, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            },
            id="all_nan_rows_excluded_partial_nan_rows_kept",
        ),
        pytest.param(
            # spatial indices outside [0, n_grid) are silently ignored — both over-bound and negative
            {
                "template": "dummy.grib",
                "start_time": datetime(2023, 1, 1, 0, 0, 0),
                "end_time": datetime(2023, 1, 1, 0, 0, 0),
                "time_freq": "6h",
                "columns": ["temperature"],
            },
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2023-01-01 05:00",
                            "2023-01-01 05:00",
                            "2023-01-01 05:00",
                        ]
                    ),
                    "spatial_index": [0, 99, -1],
                    "temperature": [10.0, 999.0, 888.0],
                }
            ),
            {
                "temperature": np.array(
                    [
                        [10.0, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan, np.nan],
                    ]
                ),
            },
            id="out_of_range_spatial_indices_ignored",
        ),
        pytest.param(
            {
                "template": "dummy.grib",
                "start_time": datetime(2023, 1, 1, 0, 0, 0),
                "end_time": datetime(2023, 1, 1, 0, 0, 0),
                "time_freq": "6h",
                "columns": ["temperature"],
            },
            pd.DataFrame(
                {
                    "date": pd.Series(dtype="datetime64[ns]"),
                    "spatial_index": pd.Series(dtype=int),
                    "temperature": pd.Series(dtype=float),
                }
            ),
            {
                "temperature": np.full((4, 4), np.nan),
            },
            id="empty_input_returns_all_nan",
        ),
    ],
)
def test_irregular_to_grid(mock_define_grid, mock_ekd_template, small_grid, config, df, expected_arrays):
    filter_fn = create_filter("irregular_to_grid", **config)
    result = filter_fn(df)

    mock_ekd_template.assert_called_once_with("file", config["template"])

    lats, lons = small_grid
    expected_times = pd.date_range(
        start=config["start_time"],
        end=config["end_time"] + pd.Timedelta(days=1),
        freq=config["time_freq"],
    )[1:]

    columns = config["columns"]
    assert len(result) == len(expected_times) * len(columns)

    for field in result:
        param = field.metadata("param")
        vdt = field.metadata("valid_datetime")
        field_lats, field_lons = field.grid_points()

        assert param in columns
        assert vdt in expected_times
        np.testing.assert_array_equal(field_lats, lats)
        np.testing.assert_array_equal(field_lons, lons)

        t_idx = expected_times.get_loc(vdt)
        np.testing.assert_array_equal(field.to_numpy(), expected_arrays[param][t_idx])


def test_missing_column_raises(mock_ekd_template):
    config = {
        "template": "dummy.grib",
        "start_time": datetime(2023, 1, 1),
        "end_time": datetime(2023, 1, 1),
        "time_freq": "6h",
        "columns": ["temperature", "missing_col"],
    }
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01 05:00"]),
            "spatial_index": [0],
            "temperature": [10.0],
        }
    )
    filter = create_filter("irregular_to_grid", **config)
    with pytest.raises(ValueError):
        filter(df)


def test_fill_grids():
    # don't usually test private methods – make an exception to allow for future optimisation
    # (remove test in future if implementation changes and this is longer useful)
    t_idx, n_spatial = 2, 4
    grids = {"temperature": np.full((t_idx, n_spatial), np.nan)}
    df_nearest = pd.DataFrame({"spatial_index": [0, 2], "temperature": [10.0, 30.0]})
    # note: grids arrays are mutated in place
    IrregularToGrid._fill_grids(grids, df_nearest, ["temperature"], n_spatial, t_idx - 1)

    # only last time index is filled in
    expected = np.array([[np.nan, np.nan, np.nan, np.nan], [10.0, np.nan, 30.0, np.nan]])
    np.testing.assert_array_equal(grids["temperature"], expected)


@pytest.mark.parametrize(
    "target_time, time_freq, input_df, expected",
    [
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:00"]),
                    "spatial_index": [0],
                    "temperature": [10.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:00"]),
                    "spatial_index": [0],
                    "temperature": [10.0],
                },
                index=pd.Index([0]),
            ),
            id="observation_inside_window_returns_data",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 06:01"]),  # exactly at target + 1min (inclusive upper bound)
                    "spatial_index": [0],
                    "temperature": [10.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 06:01"]),
                    "spatial_index": [0],
                    "temperature": [10.0],
                },
                index=pd.Index([0]),
            ),
            id="observation_at_upper_boundary_included",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 00:00"]),  # exactly at target - freq (exclusive lower bound)
                    "spatial_index": [0],
                    "temperature": [10.0],
                }
            ),
            None,
            id="observation_at_lower_boundary_excluded",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 06:02"]),  # after target + 1min
                    "spatial_index": [0],
                    "temperature": [10.0],
                }
            ),
            None,
            id="observation_after_window_returns_none",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:00"]),
                    "spatial_index": [0],
                    "temperature": [np.nan],
                }
            ),
            None,
            id="all_nan_columns_filtered_returns_none",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            "6h",
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:00", "2023-01-01 05:30"]),
                    "spatial_index": [0, 1],
                    "temperature": [np.nan, 20.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:30"]),
                    "spatial_index": [1],
                    "temperature": [20.0],
                },
                index=pd.Index([1]),
            ),
            id="all_nan_rows_filtered_partial_data_returned",
        ),
    ],
)
def test_select_window(target_time, time_freq, input_df, expected):
    # note: window is defined backwards from target_time (target_time - time_freq, target_time + 1 min)
    result = IrregularToGrid.select_window(input_df, target_time, time_freq, ["temperature"])
    if expected is None:
        assert result is None
    else:
        pd.testing.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "target_time, input_df, expected",
    [
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 06:00"]),
                    "spatial_index": [0],
                    "temperature": [42.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 06:00"]),
                    "spatial_index": [0],
                    "temperature": [42.0],
                    "time_diff": [pd.Timedelta("00:00:00")],
                },
                index=pd.Index([0]),
            ),
            id="single_spatial_index_exact_match",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 03:00", "2023-01-01 05:50", "2023-01-01 05:00"]),
                    "spatial_index": [0, 0, 0],
                    "temperature": [100.0, 200.0, 300.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:50"]),
                    "spatial_index": [0],
                    "temperature": [200.0],
                    "time_diff": [pd.Timedelta("00:10:00")],
                },
                index=pd.Index([1]),
            ),
            id="single_spatial_index_selects_nearest",
        ),
        pytest.param(
            datetime(2023, 1, 1, 6, 0),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2023-01-01 03:00", "2023-01-01 05:50", "2023-01-01 06:05", "2023-01-01 06:10"]
                    ),
                    "spatial_index": [0, 0, 1, 1],
                    "temperature": [100.0, 200.0, 300.0, 400.0],
                }
            ),
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01 05:50", "2023-01-01 06:05"]),
                    "spatial_index": [0, 1],
                    "temperature": [200.0, 300.0],
                    "time_diff": [pd.Timedelta("00:10:00"), pd.Timedelta("00:05:00")],
                },
                index=pd.Index([1, 2]),
            ),
            id="multiple_spatial_indices_selects_nearest",
        ),
    ],
)
def test_get_nearest_obs(target_time, input_df, expected):
    result = IrregularToGrid.get_nearest_obs(input_df, target_time)
    pd.testing.assert_frame_equal(result, expected, check_like=True)
