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


def test_lookup_o96() -> None:
    """Test the grids function for the 'o96' grid."""
    x = lookup("o96")
    assert x["latitudes"].mean() == pytest.approx(0.0)
    assert x["longitudes"].mean() == pytest.approx(179.14285714285714)
    assert x["latitudes"].shape == (40320,)
    assert x["longitudes"].shape == (40320,)
    assert x["latitudes"][31415] == pytest.approx(-31.324557701757268)
    assert x["longitudes"][31415] == pytest.approx(224.32835820895522)


def _regular_ll(nlat: int = 4, nlon: int = 8, lon_360: bool = True):
    """Helper: flattened regular lat/lon grid, N->S, lon starting at 0."""
    lats = np.repeat(np.linspace(90, -90, nlat), nlon)
    lons = np.tile(np.arange(nlon) * 360.0 / nlon, nlat)
    if not lon_360:
        lons = np.where(lons >= 180.0, lons - 360.0, lons)
    return lats, lons


def test_latlon_grid_identical_ordering() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    g = LatLonGrid.from_latlon(*_regular_ll())
    assert g.same_ordering_as(LatLonGrid.from_latlon(*_regular_ll()))
    np.testing.assert_array_equal(g.permutation_to(LatLonGrid.from_latlon(*_regular_ll())), np.arange(g.size))


def test_latlon_grid_reversed_latitude() -> None:
    from anemoi.transform.grids.latlon import GridComparison
    from anemoi.transform.grids.latlon import LatLonGrid

    lats, lons = _regular_ll()
    g1 = LatLonGrid.from_latlon(lats, lons)
    g2 = LatLonGrid.from_latlon(lats[::-1], lons[::-1])  # S->N ordering

    assert not g1.same_ordering_as(g2)
    assert g1.same_points_as(g2)
    result = g1.compare(g2)
    assert result.kind is GridComparison.REORDERABLE
    np.testing.assert_array_equal(result.permutation, np.arange(g1.size)[::-1])


def test_latlon_grid_longitude_convention() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    g360 = LatLonGrid.from_latlon(*_regular_ll(lon_360=True))
    g180 = LatLonGrid.from_latlon(*_regular_ll(lon_360=False))
    # Same xyz sequence despite different stored longitudes
    assert g360.same_ordering_as(g180)


def test_latlon_grid_permutation_data_outcome() -> None:
    """Reordering data must land values on the right coordinates."""
    from anemoi.transform.grids.latlon import LatLonGrid

    lats, lons = _regular_ll()
    g1 = LatLonGrid.from_latlon(lats, lons)
    g2 = LatLonGrid.from_latlon(lats[::-1], lons[::-1])

    # Data on g2's ordering: value = index of the point in g1's ordering
    data_on_g2 = np.arange(g1.size)[::-1].astype(float)
    p = g1.permutation_to(g2)
    data_on_g1 = data_on_g2[p]
    np.testing.assert_array_equal(data_on_g1, np.arange(g1.size).astype(float))


def test_latlon_grid_different_resolution() -> None:
    from anemoi.transform.grids.latlon import GridComparison
    from anemoi.transform.grids.latlon import GridMismatchError
    from anemoi.transform.grids.latlon import LatLonGrid

    g1 = LatLonGrid.from_latlon(*_regular_ll(nlat=4, nlon=8))
    g2 = LatLonGrid.from_latlon(*_regular_ll(nlat=8, nlon=16))
    assert not g1.same_points_as(g2)
    assert g1.compare(g2).kind is GridComparison.DIFFERENT
    with pytest.raises(GridMismatchError):
        g1.permutation_to(g2)


def test_latlon_grid_non_bijective() -> None:
    """Same size, shifted grid: nearest mapping is not a bijection."""
    from anemoi.transform.grids.latlon import GridMismatchError
    from anemoi.transform.grids.latlon import LatLonGrid

    lats, lons = _regular_ll()
    g1 = LatLonGrid.from_latlon(lats, lons)
    g2 = LatLonGrid.from_latlon(lats, lons + 10.0)  # shifted, same count
    assert not g1.same_points_as(g2)
    with pytest.raises(GridMismatchError) as exc_info:
        g1.permutation_to(g2)
    assert exc_info.value.num_unmatched > 0


def test_latlon_grid_from_field_duck_typed() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    class FakeField:
        def grid_points(self):
            return _regular_ll()

        def metadata(self, key, default=None):
            return {"uuidOfHGrid": "abc123"}.get(key, default)

    g = LatLonGrid.from_field(FakeField())
    assert g.metadata["uuidOfHGrid"] == "abc123"
    # uuid short-circuit: same uuid implies same ordering
    g2 = LatLonGrid.from_latlon(*_regular_ll(), uuidOfHGrid="abc123")
    assert g.same_ordering_as(g2)


def test_canonical_already_canonical() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    g = LatLonGrid.from_latlon(*_regular_ll())  # helper already produces N->S, 0->360
    assert g.is_canonical()
    np.testing.assert_array_equal(g.canonical_permutation(), np.arange(g.size))


def test_canonical_reversed_latitude() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    lats, lons = _regular_ll()
    g = LatLonGrid.from_latlon(lats[::-1], lons[::-1])
    assert not g.is_canonical()
    c = g.canonical()
    assert c.is_canonical()
    np.testing.assert_allclose(c.latitudes, lats)
    np.testing.assert_allclose(c.longitudes, lons)


def test_canonical_longitude_convention_normalized() -> None:
    from anemoi.transform.grids.latlon import LatLonGrid

    g = LatLonGrid.from_latlon(*_regular_ll(lon_360=False))  # -180..180 input
    c = g.canonical()
    assert np.all((c.longitudes >= 0) & (c.longitudes < 360))  # output in [0, 360)
    assert c.is_canonical()


def test_canonical_flip_and_shift() -> None:
    """S->N ordering AND -180..180 convention together."""
    from anemoi.transform.grids.latlon import LatLonGrid

    lats, lons = _regular_ll(lon_360=False)
    g = LatLonGrid.from_latlon(lats[::-1], lons[::-1])
    c = g.canonical()
    expected_lats, expected_lons = _regular_ll(lon_360=True)
    np.testing.assert_allclose(c.latitudes, expected_lats)
    np.testing.assert_allclose(c.longitudes, expected_lons)


def test_canonical_reshape_regular_ll() -> None:
    """Canonical order must be row-major (nlat, nlon) for regular_ll."""
    from anemoi.transform.grids.latlon import LatLonGrid

    nlat, nlon = 4, 8
    lats, lons = _regular_ll(nlat=nlat, nlon=nlon)
    # scramble, then canonicalize
    rng = np.random.default_rng(0)
    scramble = rng.permutation(lats.size)
    g = LatLonGrid.from_latlon(lats[scramble], lons[scramble])
    p = g.canonical_permutation()

    data = np.arange(lats.size, dtype=float)[scramble]
    grid2d = data[p].reshape(nlat, nlon)
    lat2d = lats[scramble][p].reshape(nlat, nlon)
    lon2d = np.mod(lons[scramble][p], 360).reshape(nlat, nlon)

    # each row is one latitude, descending; each row's lons ascend
    np.testing.assert_allclose(lat2d, np.linspace(90, -90, nlat)[:, None])
    np.testing.assert_allclose(lon2d, np.arange(nlon) * 45.0)
    # data value i sits at the i-th canonical point
    np.testing.assert_array_equal(grid2d.ravel(), np.arange(nlat * nlon))


def test_canonical_unstructured() -> None:
    """Non-regular grid: canonical order still well-defined and consistent."""
    from anemoi.transform.grids.latlon import LatLonGrid

    rng = np.random.default_rng(1)
    lats = rng.uniform(-90, 90, 100)
    lons = rng.uniform(-180, 180, 100)
    g1 = LatLonGrid.from_latlon(lats, lons)
    # same points, different order and convention
    perm = rng.permutation(100)
    g2 = LatLonGrid.from_latlon(lats[perm], np.mod(lons[perm] + 180, 360))
    c1, c2 = g1.canonical(), g2.canonical()
    assert c1.same_ordering_as(c2)  # both converge to the same ordering


def test_canonical_duplicated_wrap_column() -> None:
    """Grid containing both 0.0 and 360.0 longitude keeps both points.

    np.mod maps 360.0 -> 0.0, so the two points sort adjacently in
    canonical order. They remain distinct stored points, which means a
    wrapped grid and its unwrapped equivalent are NOT the same point
    set: same_points_as fails the bijection check (two canonical
    points would map to one). This is intentional — the point sets
    genuinely differ in cardinality of distinct locations.
    """
    from anemoi.transform.grids.latlon import LatLonGrid

    nlat = 4
    # wrapped: nlon=9 columns, 0..360 inclusive (0 and 360 duplicated)
    lats_w = np.repeat(np.linspace(90, -90, nlat), 9)
    lons_w = np.tile(np.arange(9) * 45.0, nlat)  # 0,45,...,360
    wrapped = LatLonGrid.from_latlon(lats_w, lons_w)

    # unwrapped: nlon=8 columns, 0..315
    lats_u, lons_u = _regular_ll(nlat=nlat, nlon=8)
    unwrapped = LatLonGrid.from_latlon(lats_u, lons_u)

    # canonicalization keeps both wrap points, sorted adjacently
    c = wrapped.canonical()
    assert c.size == 36
    # first row: 0.0 (from 360.0) then 0.0 (from 0.0) then 45...
    row0_lons = c.longitudes[:9]
    assert row0_lons[0] == 0.0 and row0_lons[1] == 0.0

    # different sizes -> not the same point set
    assert not wrapped.same_points_as(unwrapped)

    # same-size case: wrapped grid vs itself shifted — bijection still
    # holds because both sides carry the duplicate
    assert wrapped.same_points_as(LatLonGrid.from_latlon(lats_w[::-1], lons_w[::-1]))


if __name__ == "__main__":
    from anemoi.utils.testing import run_tests

    run_tests(globals())
