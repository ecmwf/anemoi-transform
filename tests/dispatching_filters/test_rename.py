# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import pandas as pd
import pytest

from ..utils import create_dispatching_filter as create_filter


@pytest.fixture
def grib_source(test_source):
    return test_source("anemoi-datasets/create/grib-20100101.grib")


def test_rename_tabular():
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


def test_rename_field(grib_source):
    rename = create_filter(
        "rename",
        param={"z": "geopotential", "t": "temperature"},
    )
    pipeline = grib_source | rename

    for original, result in zip(grib_source, pipeline):
        assert isinstance(result, ekd.Field)
        if original.parameter.variable() == "z":
            assert result.parameter.variable() == "geopotential"
        elif original.parameter.variable() == "t":
            assert result.parameter.variable() == "temperature"
        else:
            raise RuntimeError(f"Unexpected param: {original.metadata('param')}")
