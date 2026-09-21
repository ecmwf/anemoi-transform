# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction

import pandas as pd

from anemoi.transform.filters import create_filter_by_name as create_filter


def test_filter_query():
    config = {
        "query": "col1 in [1,2] and col2 in ['a', 'b']",
    }
    df = pd.DataFrame(
        {
            "col1": [1, 2, 3, 2, 1],
            "col2": ["a", "b", "a", "d", "e"],
            "col3": [0, 1, 2, 3, 4],
        }
    )
    filter_query = create_filter("filter_query", **config)
    result = filter_query(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns)
    assert result.shape == (2, len(df.columns))

    assert result.equals(df.iloc[[0, 1]])
