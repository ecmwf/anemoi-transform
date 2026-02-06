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


def test_mask_dewpoint_temperature_only():
    config = {"mask_specific_humidity": False}
    df = pd.DataFrame(
        {
            "2d": [1.0, 2.0, 3.1, np.nan, 5.0],
            "2t": [1.0, 2.1, 3.0, 4.0, np.nan],
            "x": [1, 2, 3, 4, 5],
        }
    )
    mask_dewpoint_temperature = create_filter("mask_dewpoint_temperature", **config)
    result = mask_dewpoint_temperature(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result[["2t", "x"]].equals(df[["2t", "x"]])
    # mask 2t where 2t < 2d
    assert np.allclose(
        result["2d"].to_numpy(),
        [1.0, 2.0, np.nan, np.nan, 5.0],
        equal_nan=True,
    )


def test_mask_dewpoint_temperature_only_column_names():
    config = {
        "temperature": "obsvalue_t2m_0",
        "dewpoint_temperature": "obsvalue_td2m_0",
        "mask_specific_humidity": False,
    }
    df = pd.DataFrame(
        {
            "obsvalue_td2m_0": [1.0, 2.0, 3.1, np.nan, 5.0],
            "obsvalue_t2m_0": [1.0, 2.1, 3.0, 4.0, np.nan],
            "x": [1, 2, 3, 4, 5],
        }
    )
    mask_dewpoint_temperature = create_filter("mask_dewpoint_temperature", **config)
    result = mask_dewpoint_temperature(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result[["obsvalue_t2m_0", "x"]].equals(df[["obsvalue_t2m_0", "x"]])
    # mask 2t where 2t < 2d
    assert np.allclose(
        result["obsvalue_td2m_0"].to_numpy(),
        [1.0, 2.0, np.nan, np.nan, 5.0],
        equal_nan=True,
    )


def test_mask_dewpoint_temperature_with_specific_humidity():
    config = {
        "temperature": "my_t",
        "dewpoint_temperature": "my_td",
        "specific_humidity": "my_q",
        "mask_specific_humidity": True,
    }
    df = pd.DataFrame(
        {
            "my_td": [1.0, 2.0, 3.1, np.nan, 5.0],
            "my_t": [1.0, 2.1, 3.0, 4.0, np.nan],
            "my_q": [1, 2, 3, 4, 5],
        }
    )
    mask_dewpoint_temperature = create_filter("mask_dewpoint_temperature", **config)
    result = mask_dewpoint_temperature(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape

    assert result["my_t"].equals(df["my_t"])
    # mask td where t < td
    assert np.allclose(
        result["my_td"].to_numpy(),
        [1.0, 2.0, np.nan, np.nan, 5.0],
        equal_nan=True,
    )
    assert np.allclose(
        result["my_q"].to_numpy(),
        [1, 2, np.nan, 4, 5],
        equal_nan=True,
    )


def test_mask_dewpoint_temperature_only_missing_column():
    config = {
        "temperature": "obsvalue_t2m_0",
        "dewpoint_temperature": "obsvalue_td2m_0",
        "mask_specific_humidity": False,
    }
    df = pd.DataFrame(
        {
            # obsvalue_td2m_0 column missing
            "obsvalue_t2m_0": [1.0, 2.1, 3.0, 4.0, np.nan],
            "x": [1, 2, 3, 4, 5],
        }
    )
    mask_dewpoint_temperature = create_filter("mask_dewpoint_temperature", **config)
    with pytest.raises(ValueError):
        _ = mask_dewpoint_temperature(df.copy())


def test_mask_dewpoint_temperature_with_specific_humidity_missing_column():
    config = {
        "temperature": "obsvalue_t2m_0",
        "dewpoint_temperature": "obsvalue_td2m_0",
        "specific_humidity": "obsvalue_q2m_0",
        "mask_specific_humidity": True,
    }
    df = pd.DataFrame(
        {
            # obsvalue_q2m_0 column missing
            "obsvalue_t2m_0": [1.0, 2.1, 3.0, 4.0, np.nan],
            "obsvalue_td2m_0": [1.0, 2.1, 3.0, 4.0, np.nan],
            "x": [1, 2, 3, 4, 5],
        }
    )
    mask_dewpoint_temperature = create_filter("mask_dewpoint_temperature", **config)
    with pytest.raises(ValueError):
        _ = mask_dewpoint_temperature(df.copy())
