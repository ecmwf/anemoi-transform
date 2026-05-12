# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from anemoi.transform.filters.fields.snow_depth_m import compute_snow_depth_m


def test_compute_snow_depth_m() -> None:
    """Test the compute_snow_depth_m function.

    Tests:
    - Snow depth in metres = 1000 * sd / rsn.
    - Verifies the conversion with known values.
    """
    # sd = 0.01 m water equivalent, rsn = 200 kg/m³
    # sde = 1000 * 0.01 / 200 = 0.05 m
    snow_depth = np.array([0.01, 0.02, 0.05, 0.1])
    snow_density = np.array([200.0, 250.0, 300.0, 400.0])
    expected = 1000.0 * snow_depth / snow_density

    result = compute_snow_depth_m(snow_depth, snow_density)
    np.testing.assert_allclose(result, expected)


def test_compute_snow_depth_m_values() -> None:
    """Test with specific known values."""
    # 10 cm water equivalent with density 500 kg/m³ → 0.2 m snow depth
    snow_depth = np.array([0.1])
    snow_density = np.array([500.0])
    result = compute_snow_depth_m(snow_depth, snow_density)
    np.testing.assert_allclose(result, np.array([0.2]))


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
