# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.grids import UnstructuredGridFieldList
from anemoi.transform.grids.named import lookup

latitude_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlat = "tlat"

longitudes_url = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_extpar_0026_R03B07_G_20150805.g2"
tlon = "tlon"


@skip_if_offline
@pytest.mark.slow
def test_unstructured_from_url() -> None:
    """Test the UnstructuredGridFieldList class for loading data from URLs.

    Tests:
    - Loading latitude and longitude data from URLs.
    - Asserting the loaded data has the correct number of grid points.
    - Creating forcings from the loaded data and asserting their properties.
    """
    ds = UnstructuredGridFieldList.from_grib(latitude_url, longitudes_url, tlat, tlon)

    assert len(ds) == 1

    lats, lons = ds[0].geography.latlons(flatten=True)

    assert len(lats) == len(lons)

    forcings = ekd.from_source(
        "forcings",
        ds,
        date="2015-08-05",
        param=["cos_latitude", "sin_latitude"],
    ).to_fieldlist()

    assert len(forcings) == 2


def test_unstructured_from_values() -> None:
    """Test UnstructuredGridFieldList.from_values and its use as a forcings source.

    This mirrors how anemoi-inference builds computed forcings, and runs offline
    so it guards the earthkit-data field protocol without needing the network.
    """
    latitudes = np.array([10.0, 0.0, -10.0])
    longitudes = np.array([20.0, 30.0, 40.0])

    ds = UnstructuredGridFieldList.from_values(latitudes=latitudes, longitudes=longitudes)

    assert isinstance(ds, UnstructuredGridFieldList)
    assert len(ds) == 1

    lats, lons = ds[0].geography.latlons(flatten=True)
    np.testing.assert_array_equal(lats, latitudes)
    np.testing.assert_array_equal(lons, longitudes)

    # the field must be usable as a grid template by the forcings source
    forcings = ekd.from_source(
        "forcings",
        ds,
        date=["2015-08-05"],
        param=["cos_latitude", "sin_latitude"],
    ).to_fieldlist()

    assert len(forcings) == 2
    assert [f.parameter.variable() for f in forcings] == ["cos_latitude", "sin_latitude"]
    np.testing.assert_allclose(forcings[0].to_numpy(), np.cos(np.deg2rad(latitudes)))
    np.testing.assert_allclose(forcings[1].to_numpy(), np.sin(np.deg2rad(latitudes)))


def test_unstructured_from_values_accepts_lists() -> None:
    """Lists are accepted and converted to arrays."""
    ds = UnstructuredGridFieldList.from_values(latitudes=[10.0, 0.0], longitudes=[20.0, 30.0])

    lats, lons = ds[0].geography.latlons(flatten=True)
    np.testing.assert_array_equal(lats, np.array([10.0, 0.0]))
    np.testing.assert_array_equal(lons, np.array([20.0, 30.0]))


def test_unstructured_multiple_dates() -> None:
    """The forcings source returns one field per parameter and date.

    This is the assertion anemoi-inference makes in ComputedForcings.
    """
    ds = UnstructuredGridFieldList.from_values(latitudes=[10.0, 0.0], longitudes=[20.0, 30.0])

    dates = ["2015-08-05", "2015-08-06"]
    params = ["cos_julian_day", "insolation"]
    forcings = ekd.from_source("forcings", ds, date=dates, param=params).to_fieldlist()

    assert len(forcings) == len(params) * len(dates)


def test_lookup_o96() -> None:
    """Test the grids function for the 'o96' grid."""
    x = lookup("o96")
    assert x["latitudes"].mean() == pytest.approx(0.0)
    assert x["longitudes"].mean() == pytest.approx(179.14285714285714)
    assert x["latitudes"].shape == (40320,)
    assert x["longitudes"].shape == (40320,)
    assert x["latitudes"][31415] == pytest.approx(-31.324557701757268)
    assert x["longitudes"][31415] == pytest.approx(224.32835820895522)


if __name__ == "__main__":
    from anemoi.utils.testing import run_tests

    run_tests(globals())


def test_new_field_from_grid() -> None:
    """Test that new_field_from_grid applies a Grid's latitudes and longitudes."""
    from anemoi.transform.fields import new_field_from_grid
    from anemoi.transform.grids import Grid

    latitudes = np.array([1.0, 2.0])
    longitudes = np.array([3.0, 4.0])

    class _FakeGrid(Grid):
        def latlon(self):
            return latitudes, longitudes

    template = ekd.from_source(
        "list-of-dicts",
        [
            {
                "parameter": {"variable": "2t"},
                "data": {"values": np.zeros(2)},
                "geography": {"latitudes": np.zeros(2), "longitudes": np.zeros(2)},
            }
        ],
    ).to_fieldlist()[0]

    field = new_field_from_grid(template, _FakeGrid())

    lats, lons = field.geography.latlons(flatten=True)
    np.testing.assert_allclose(lats, latitudes)
    np.testing.assert_allclose(lons, longitudes)
