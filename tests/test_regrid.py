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
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry

LOG = logging.getLogger(__name__)


def compare_npz_files(file1, file2):
    import numpy as np

    data1 = np.load(file1)
    data2 = np.load(file2)

    assert set(data1.keys()) == set(
        data2.keys()
    ), f"Keys in NPZ files do not match {set(data1.keys())} and {set(data2.keys())}"

    for key in data1.keys():
        try:
            assert (data1[key] == data2[key]).all(), f"Data for key {key} does not match between {file1} and {file2}"
        except Exception as e:
            LOG.error(f"Error comparing key {key} :between {file1} and {file2}: {e}")
            raise


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
def test_regrid_mask():

    mask = get_test_data("anemoi-transform/filters/regrid/ea-over-rr-mask.npz")
    mask = "ea-over-rr-mask.npz"

    era5 = source_registry.create(
        "testing",
        dataset="anemoi-transform/filters/regrid/2t-ea.grib",
    )
    regrid = filter_registry.create("regrid", mask=mask)
    for _ in era5 | regrid:
        pass


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    logging.basicConfig(level=logging.INFO)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
