# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import io

import numpy as np
import pytest
from anemoi.utils.testing import cli_testing
from anemoi.utils.testing import skip_if_missing_command
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import create_filter_by_name as create_filter

from ..utils import compare_npz_files

# A regular lat/lon grid, for which geography.latlons() returns 2D arrays.
LATITUDES = [10.0, 0.0, -10.0]
LONGITUDES = [20.0, 40.0, 60.0, 80.0]
VALUES = np.arange(12.0).reshape(3, 4)


@skip_if_offline
@skip_if_missing_command("mir")
@pytest.mark.slow
def test_make_regrid_matrix(get_test_data):
    era5 = get_test_data("anemoi-transform/filters/regrid/2t-ea.grib")
    carra = get_test_data("anemoi-transform/filters/regrid/2t-rr.grib")
    mask = get_test_data("anemoi-transform/filters/regrid/ea-to-rr-matrix.npz")

    cli_testing(
        "anemoi-transform",
        "make-regrid-file",
        "mir-matrix",
        "--source-grid",
        era5,
        "--target-grid",
        carra,
        "--output",
        "ea-to-rr-matrix.npz",
    )

    compare_npz_files(mask, "ea-to-rr-matrix.npz")


@skip_if_offline
def test_regrid_matrix(get_test_data, test_source):
    matrix = get_test_data("anemoi-transform/filters/regrid/ea-to-rr-matrix.npz")

    era5 = test_source("anemoi-transform/filters/regrid/2t-ea.grib")
    regrid = create_filter("regrid", matrix=matrix)
    for _ in era5 | regrid:
        pass


@skip_if_offline
@pytest.mark.slow
def test_make_regrid_mask(get_test_data):
    era5 = get_test_data("anemoi-transform/filters/regrid/2t-ea.grib")
    carra = get_test_data("anemoi-transform/filters/regrid/2t-rr.grib")
    mask = get_test_data("anemoi-transform/filters/regrid/ea-over-rr-mask.npz")

    cli_testing(
        "anemoi-transform",
        "make-regrid-file",
        "global-on-lam-mask",
        "--global-grid",
        era5,
        "--lam-grid",
        carra,
        "--output",
        "ea-over-rr-mask.npz",
    )

    compare_npz_files(mask, "ea-over-rr-mask.npz")


@skip_if_offline
def test_regrid_ekd(test_source):
    era5 = test_source("anemoi-transform/filters/regrid/2t-ea.grib")
    regrid = create_filter("regrid", in_grid="N320", out_grid=[0.25, 0.25])
    for _ in era5 | regrid:
        pass


@skip_if_offline
def test_regrid_nearest(test_source):
    era5 = test_source("anemoi-transform/filters/regrid/2t-ea.grib")
    regrid = create_filter("regrid", in_grid="N320", out_grid="O96", method="nearest")
    for _ in era5 | regrid:
        pass


@skip_if_offline
def test_regrid_mask(get_test_data, test_source):
    mask = get_test_data("anemoi-transform/filters/regrid/ea-over-rr-mask.npz")

    era5 = test_source("anemoi-transform/filters/regrid/2t-ea.grib")
    regrid = create_filter("regrid", mask=mask)
    for _ in era5 | regrid:
        pass


def test_regrid_mask_regular_grid(test_source):
    # Regression test: on a regular grid geography.latlons() is 2D, but the mask
    # is built against the flattened values, so the filter must flatten the
    # coordinates before applying it.
    mask = np.array([False, True, False, True, True, False] * 2)

    # the filter loads the mask with np.load, which also accepts a file-like object
    mask_file = io.BytesIO()
    np.savez(mask_file, mask=mask)
    mask_file.seek(0)

    source = test_source(
        [
            {
                "parameter.variable": "2t",
                "geography.distinct_latitudes": LATITUDES,
                "geography.distinct_longitudes": LONGITUDES,
                "time.valid_datetime": "2018-08-01T09:00:00Z",
                "data.values": VALUES,
            }
        ]
    )

    regrid = create_filter("regrid", mask=mask_file)
    (result,) = list(source | regrid)

    # the masked field keeps the selected points only, as flat arrays
    np.testing.assert_allclose(result.to_numpy(flatten=True), VALUES.flatten()[mask])

    expected_lats, expected_lons = np.meshgrid(LATITUDES, LONGITUDES, indexing="ij")
    result_lats, result_lons = result.geography.latlons(flatten=True)
    np.testing.assert_allclose(result_lats, expected_lats.flatten()[mask])
    np.testing.assert_allclose(result_lons, expected_lons.flatten()[mask])


if __name__ == "__main__":
    from anemoi.utils.testing import run_tests

    run_tests(globals())
