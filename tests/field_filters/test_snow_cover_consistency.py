# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.transform.filters import filter_registry

from ..utils import collect_fields_by_param

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20.0, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

# Each cell exercises a specific rule or no-change case (snowc clipped to [0, 1]):
#   [0,0]: sd=0,  sc=50   → rule 1 (sc>0 & sd=0):  sc=0
#   [0,1]: sd=10, sc=0    → rule 2 (sd>5 & sc=0):  sc=10*0.15=1.5, clipped→1.0
#   [1,0]: sd=3,  sc=0    → no rule (sd<=5, sc=0): sc=0
#   [1,1]: sd=10, sc=0.8  → no rule (sd>5, sc>0):  sc=0.8
#   [2,0]: sd=0,  sc=0    → no rule (both 0):      sc=0
#   [2,1]: sd=3,  sc=2.0  → clipped to 1.0
SD_VALUES = np.array([[0.0, 10.0], [3.0, 10.0], [0.0, 3.0]])
SC_VALUES = np.array([[50.0, 0.0], [0.0, 0.8], [0.0, 2.0]])
SC_EXPECTED = np.array([[0.0, 1.0], [0.0, 0.8], [0.0, 1.0]])


@pytest.fixture
def source(test_source):
    field_specs = [
        {"param": "sd", "values": SD_VALUES.copy(), **MOCK_FIELD_METADATA},
        {"param": "snowc", "values": SC_VALUES.copy(), **MOCK_FIELD_METADATA},
    ]
    return test_source(field_specs)


def test_snow_cover_consistency(source):
    filt = filter_registry.create("snow_cover_consistency")
    pipeline = source | filt
    output = collect_fields_by_param(pipeline)

    assert set(output) == {"snowc", "sd"}

    assert len(output["snowc"]) == 1
    np.testing.assert_allclose(output["snowc"][0].to_numpy(), SC_EXPECTED)

    assert len(output["sd"]) == 1
    np.testing.assert_allclose(output["sd"][0].to_numpy(), SD_VALUES)


@pytest.mark.parametrize(
    "sd, sc_in, sc_out",
    [
        # Rule 1: sc > 0 and sd = 0  →  sc = 0
        (0.0, 50.0, 0.0),
        (0.0, 0.001, 0.0),
        # Rule 2: sd > 5 and sc = 0  →  sc = sd * 0.15, clipped at 1
        (10.0, 0.0, 1.0),  # sd*0.15=1.5, clipped to 1
        (6.0, 0.0, 0.9),
        # No rule applies, clipping still enforced
        (5.0, 0.0, 0.0),  # sd not strictly > 5
        (10.0, 2.0, 1.0),  # sc > 1, clipped to 1
        (3.0, 0.5, 0.5),  # sd > 0 and sc > 0, sd <= 5
        (0.0, 0.0, 0.0),  # both zero
    ],
)
def test_snow_cover_consistency_scalar(test_source, sd, sc_in, sc_out):
    field_specs = [
        {
            "param": "sd",
            "values": np.array([[sd]]),
            "latitudes": [0.0],
            "longitudes": [0.0],
            "valid_datetime": "2018-08-01T09:00:00Z",
        },
        {
            "param": "snowc",
            "values": np.array([[sc_in]]),
            "latitudes": [0.0],
            "longitudes": [0.0],
            "valid_datetime": "2018-08-01T09:00:00Z",
        },
    ]
    source = test_source(field_specs)
    filt = filter_registry.create("snow_cover_consistency")
    pipeline = source | filt
    output = collect_fields_by_param(pipeline)

    result = output["snowc"][0].to_numpy().item()
    assert result == pytest.approx(sc_out), f"sd={sd}, sc_in={sc_in}: expected {sc_out}, got {result}"
