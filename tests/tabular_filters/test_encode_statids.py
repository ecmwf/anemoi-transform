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


def test_encode_statids():
    config = {}
    df = pd.DataFrame(
        {
            "statid": ["here", "there", "1001"],
        }
    )
    encode_statids = create_filter("encode_statids", **config)
    result = encode_statids(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape
    expected = pd.DataFrame(
        {
            "statid": [812282, 49521146, 46657],
        }
    )
    assert result["statid"].equals(expected["statid"])


def test_encode_statids_with_config():
    config = {"station_id": "mystatid"}
    df = pd.DataFrame(
        {
            "mystatid": ["here", "there", "1001"],
        }
    )
    encode_statids = create_filter("encode_statids", **config)
    result = encode_statids(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == df.shape
    expected = pd.DataFrame(
        {
            "mystatid": [812282, 49521146, 46657],
        }
    )
    assert result["mystatid"].equals(expected["mystatid"])


def test_encode_statids_missing_column():
    config = {"station_id": "statid"}
    df = pd.DataFrame(
        {
            # statid column missing
            "foo": ["here", "there", "1001"],
        }
    )
    encode_statids = create_filter("encode_statids", **config)
    with pytest.raises(ValueError):
        _ = encode_statids(df.copy())
