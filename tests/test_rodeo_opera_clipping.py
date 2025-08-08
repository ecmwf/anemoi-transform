# (C) Copyright 2025 Anemoi contributors.
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

from .utils import collect_fields_by_param

MAX_TP = 12.5

MOCK_FIELD_METADATA = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

expected_tp_values = np.array(
    [
        [0.0, 0.0],
        [0.001, np.nan],
        [0.0125, np.nan],
    ]
)
expected_qi_values = np.array(
    [
        [0, 0.5],
        [0.2, np.nan],
        [1.0, np.nan],
    ]
)


@pytest.fixture
def rodeo_opera_source(test_source):
    """Create mock Rodeo OPERA dataset with tp,and qi fields."""
    tp_values = np.array(
        [
            [-5.0, 0.0],
            [1.0, np.nan],
            [20.0, np.nan],
        ]
    )
    qi_values = np.array(
        [
            [-1.0, 0.5],
            [0.2, np.nan],
            [1.2, np.nan],
        ]
    )

    SPEC = [
        {"param": "tp", "values": tp_values, **MOCK_FIELD_METADATA},
        {"param": "qi", "values": qi_values, **MOCK_FIELD_METADATA},
    ]
    return test_source(SPEC)


def test_rodeo_opera_clipping(rodeo_opera_source):
    preprocessing = filter_registry.create("rodeo_opera_clipping", max_total_precipitation=MAX_TP)
    pipeline = rodeo_opera_source | preprocessing

    output_fields = collect_fields_by_param(pipeline)

    assert set(output_fields) == {"tp", "qi"}
    assert len(output_fields["tp"]) == 1
    assert len(output_fields["qi"]) == 1

    tp = output_fields["tp"][0].to_numpy()
    qi = output_fields["qi"][0].to_numpy()

    assert np.allclose(tp, expected_tp_values, equal_nan=True)
    # Sanity check: NaNs in tp must match NaNs in qi
    assert np.allclose(qi, expected_qi_values, equal_nan=True)

    assert np.isnan(tp).sum() == np.isnan(qi).sum()
    assert np.nanmax(tp) <= MAX_TP
    assert np.nanmin(tp) >= 0.0
    assert np.nanmax(qi) <= 1
    assert np.nanmin(qi) >= 0.0


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
