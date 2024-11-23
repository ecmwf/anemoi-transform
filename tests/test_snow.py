# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.transform.filters.snow_cover import compute_snow_cover


@pytest.mark.skip("Test not implemented")
def test_snow_cover():
    snow_depth = np.array([1.0, 2.0, 3.0])
    snow_density = np.array([0.1, 0.2, 0.3])
    expected_snow_cover = np.array([0.1, 0.4, 0.9])
    snow_cover = compute_snow_cover(snow_depth, snow_density)
    np.testing.assert_allclose(snow_cover, expected_snow_cover)
