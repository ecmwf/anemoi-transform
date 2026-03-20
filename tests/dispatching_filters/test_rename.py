# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd
import pytest

from anemoi.transform.fields import WrappedField
from anemoi.transform.filters import dispatching_filter_registry as filter_registry


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
    rename = filter_registry.create("rename", **config)
    result = rename(df.copy())
    assert isinstance(result, pd.DataFrame)


def test_rename_field(grib_source):
    rename = filter_registry.create(
        "rename",
        param={"z": "geopotential", "t": "temperature"},
    )
    pipeline = grib_source | rename

    for original, result in zip(grib_source, pipeline):
        assert isinstance(result, WrappedField)
        if original.metadata("param") == "z":
            assert result.metadata("param") == "geopotential"
        elif original.metadata("param") == "t":
            assert result.metadata("param") == "temperature"
        else:
            raise RuntimeError(f"Unexpected param: {original.metadata('param')}")
