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


def test_add_healpix():
    config = {
        "nside": 16,
    }
    df = pd.DataFrame(
        {
            "latitude": [-89.9, -89.9, -89.9, 0.0, 0.0, 0.0, 89.9, 89.9, 89.9],
            "longitude": [0.1, 180.0, 359.9, 0.1, 180.0, 359.9, 0.1, 180.0, 359.9],
        }
    )
    add_healpix = create_filter("add_healpix", **config)
    result = add_healpix(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + (f"healpix_idx_{config['nside']}",)
    assert result.shape == (len(df), len(df.columns) + 1)

    assert result[["latitude", "longitude"]].equals(df[["latitude", "longitude"]])

    expected_result = pd.Series(
        [2048, 2560, 2816, 1130, 1642, 1173, 255, 767, 1023],
        name=f"healpix_idx_{config['nside']}",
    )
    assert result[f"healpix_idx_{config['nside']}"].equals(expected_result)


def test_add_healpix_bad_nside():
    config = {
        "nside": -1,
    }
    with pytest.raises(ValueError):
        _ = create_filter("add_healpix", **config)
