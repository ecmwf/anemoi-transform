# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd

from anemoi.transform.grids import UnstructuredGridFieldList
from anemoi.transform.grids.named import lookup

latitude_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlat = "tlat"

longitudes_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlon = "tlon"


def do_not_test_unstructured_from_url() -> None:
    """Test the UnstructuredGridFieldList class for loading data from URLs.

    Tests:
    - Loading latitude and longitude data from URLs.
    - Asserting the loaded data has the correct number of grid points.
    - Creating forcings from the loaded data and asserting their properties.
    """
    ds = UnstructuredGridFieldList.from_grib(latitude_url, longitudes_url, tlat, tlon)

    assert len(ds) == 1

    lats, lons = ds[0].grid_points()

    assert len(lats) == len(lons)

    forcings = ekd.from_source(
        "forcings",
        ds,
        date="2015-08-05",
        param=["cos_latitude", "sin_latitude"],
    )

    assert len(forcings) == 2


def test_o96() -> None:
    """Test the grids function for the 'o96' grid."""
    x = lookup("o96")
    assert x["latitudes"].mean() == 0.0
    assert x["longitudes"].mean() == 179.14285714285714
    assert x["latitudes"].shape == (40320,)
    assert x["longitudes"].shape == (40320,)
    assert x["latitudes"][31415] == -31.324557701757268
    assert x["longitudes"][31415] == 224.32835820895522


if __name__ == "__main__":
    from anemoi.utils.testing import run_tests

    run_tests(globals())
