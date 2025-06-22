# (C) Copyright 2025 Anemoi contributors.
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

import logging

from anemoi.utils.testing import cli_testing
from anemoi.utils.testing import get_test_data
from anemoi.utils.testing import skip_if_missing_command
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry
from anemoi.transform.testing import compare_npz_files

LOG = logging.getLogger(__name__)


@skip_if_offline
@skip_if_missing_command("mir")
def test_make_regrid_matrix():

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
def test_regrid_matrix():

    matrix = get_test_data("anemoi-transform/filters/regrid/ea-to-rr-matrix.npz")

    era5 = source_registry.create(
        "testing",
        dataset="anemoi-transform/filters/regrid/2t-ea.grib",
    )
    regrid = filter_registry.create("regrid", matrix=matrix)
    for _ in era5 | regrid:
        pass


@skip_if_offline
def test_make_regrid_mask():

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
def test_regrid_ekd():

    era5 = source_registry.create(
        "testing",
        dataset="anemoi-transform/filters/regrid/2t-ea.grib",
    )
    regrid = filter_registry.create("regrid", in_grid="N320", out_grid=[0.25, 0.25])
    for _ in era5 | regrid:
        pass


@skip_if_offline
def test_regrid_nearest():

    era5 = source_registry.create(
        "testing",
        dataset="anemoi-transform/filters/regrid/2t-ea.grib",
    )
    regrid = filter_registry.create("regrid", in_grid="N320", out_grid="O96", method="nearest")
    for _ in era5 | regrid:
        pass


@skip_if_offline
def test_regrid_mask():

    mask = get_test_data("anemoi-transform/filters/regrid/ea-over-rr-mask.npz")

    era5 = source_registry.create(
        "testing",
        dataset="anemoi-transform/filters/regrid/2t-ea.grib",
    )
    regrid = filter_registry.create("regrid", mask=mask)
    for _ in era5 | regrid:
        pass


if __name__ == "__main__":
    test_regrid_nearest()
    # from anemoi.utils.testing import run_tests
    # run_tests(globals())
