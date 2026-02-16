# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from unittest import mock

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi.transform.filters import filter_registry

from .utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

SNOW_DEPTH_VALUES = np.array([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])
GLACIER_MASK = np.array([[0, 0], [0, 1], [1, 1]])


@pytest.fixture
def snow_depth_source(test_source):
    SNOW_DEPTH_SPEC = [{"param": "sd", "values": SNOW_DEPTH_VALUES.copy(), **MOCK_FIELD_METADATA}]
    return test_source(SNOW_DEPTH_SPEC)


@pytest.fixture
def mock_mask():
    field = {"param": "glacier_mask", "values": GLACIER_MASK.copy(), **MOCK_FIELD_METADATA}
    return ekd.from_source("list-of-dicts", [field])


def test_glacier_mask(snow_depth_source, mock_mask):
    with mock.patch("anemoi.transform.filters.glacier_mask.ekd.from_source") as mock_earthkit:
        mock_earthkit.return_value = mock_mask

        glacier_mask = filter_registry.create("glacier_mask", glacier_mask="glacier_mask.grib")
        mock_earthkit.assert_called_once_with("file", "glacier_mask.grib")

    pipeline = snow_depth_source | glacier_mask

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"sd_masked"}
    assert len(output_fields["sd_masked"]) == 1
    expected_value = np.ma.array(SNOW_DEPTH_VALUES, mask=GLACIER_MASK.astype(bool)).filled(np.nan)
    assert np.allclose(output_fields["sd_masked"][0].to_numpy(), expected_value, equal_nan=True)
