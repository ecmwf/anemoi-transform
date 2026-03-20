# (C) Copyright 2026- Anemoi contributors.
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

from ..utils import assert_fields_equal
from ..utils import collect_fields_by_param
from ..utils import create_dispatching_filter as create_filter

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

OROG_VALUES = np.array([[243.87788459, 1892.45371246], [427.80215359, 156.92873391], [2167.93458212, 338.15794671]])


@pytest.fixture
def orog_source(test_source):
    OROG_SPEC = [{"param": "orog", "values": OROG_VALUES, **MOCK_FIELD_METADATA}]
    return test_source(OROG_SPEC)


def test_orog_to_z_fields(orog_source):
    orog_to_z = create_filter("orog_to_z")
    z_to_orog = create_filter("z_to_orog")
    z_source = orog_source | orog_to_z
    pipeline = z_source | z_to_orog

    input_fields = collect_fields_by_param(orog_source)
    intermediate_fields = collect_fields_by_param(z_source)
    output_fields = collect_fields_by_param(pipeline)

    assert set(input_fields) == {"orog"}
    assert len(input_fields["orog"]) == 1
    assert set(intermediate_fields) == {"z"}
    assert len(intermediate_fields["z"]) == 1
    assert set(output_fields) == {"orog"}
    assert len(output_fields["orog"]) == 1
    assert_fields_equal(input_fields["orog"][0], output_fields["orog"][0])


def test_geopotential_to_height_tabular():
    config = {
        "geopotential": "z",
        "height": "height",
    }
    df = pd.DataFrame(
        {
            "z": [1.0, 2.0, 3.0, 4.0],
        }
    )
    geopotential_to_height = create_filter("geopotential_to_height", **config)
    result = geopotential_to_height(df.copy())
    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == tuple(df.columns) + ("height",)
    assert result.shape == (df.shape[0], df.shape[1] + 1)
    assert result["height"].equals(df["z"] / 9.80665)
