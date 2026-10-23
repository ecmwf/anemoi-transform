# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.transform.filters.fields.fraction_of_snow_cover import compute_fraction_of_snow_cover


@pytest.mark.skip("Test not implemented")
def test_fraction_of_snow_cover() -> None:
    """Test the compute_fraction_of_snow_cover function.

    Tests:
    - Computing fraction of snow cover from given snow depth and snow density arrays.
    - Asserting the computed fraction of snow cover matches the expected values.
    """
    snow_depth: np.ndarray = np.array([1.0, 2.0, 3.0])
    snow_density: np.ndarray = np.array([0.1, 0.2, 0.3])
    expected_fscov: np.ndarray = np.array([0.1, 0.4, 0.9])
    fscov: np.ndarray = compute_fraction_of_snow_cover(snow_depth, snow_density)
    np.testing.assert_allclose(fscov, expected_fscov)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
