# (C) Copyright 2026 Anemoi contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest

from anemoi.transform.spatial import cutout_mask


@pytest.mark.parametrize("cropping_distance", [1.0, 3.0, 5.0])
def test_cutout_mask_with_max_distance(cropping_distance: float):
    """Test cutout_mask with max_distance_km parameter.

    The results should be independent of the cropping_distance parameter.
    """
    # Create a LAM region
    lam_lat_range = np.linspace(44.0, 46.0, 11)
    lam_lon_range = np.linspace(0.0, 2.0, 11)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid with points at varying distances
    global_lats = np.array([43.1, 44.0, 45.0, 45.5, 46.0, 50.0])
    global_lons = np.array([359.1, 359.5, 0.0, 1.0, 2.0, 0.0])

    # Apply mask with max_distance_km to exclude far points
    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        cropping_distance=cropping_distance,
        max_distance_km=250.0,  # 250 km limit
    )

    # The first point at lat=43.1 should be included (distance in [0, max_distance_km])
    # The next 4 points should be excluded (inside)
    # The last point at lat=50.0 should be excluded (too far)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == global_lats.shape
    assert np.array_equal(mask, np.array([True, False, False, False, False, False]))


def test_cutout_mask_with_min_distance():
    """Test cutout_mask with both min_distance_km."""
    # Create a LAM region
    lam_lat_range = np.linspace(44.0, 46.0, 11)
    lam_lon_range = np.linspace(0.0, 2.0, 11)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid
    global_lats = np.array([44.0, 45.0, 46.0, 46.1, 47.5])
    global_lons = np.array([0.0, 1.0, 2.0, -0.1, -1.5])

    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        min_distance_km=100.0,
    )

    # The first 3 points should be excluded (inside)
    # The 4th point at lat=46.1 should be excluded (too close)
    # The last point at lat=47.5 should be included
    assert isinstance(mask, np.ndarray)
    assert mask.shape == global_lats.shape
    assert np.array_equal(mask, np.array([False, False, False, False, True]))


def test_cutout_mask_array_shapes():
    """Test that input arrays must be 1D."""
    lam_lats = np.array([[45.0, 45.0], [46.0, 46.0]])
    lam_lons = np.array([[0.0, 1.0], [0.0, 1.0]])
    global_lats = np.array([45.0])
    global_lons = np.array([0.0])

    # Should raise assertion error due to 2D arrays
    with pytest.raises(AssertionError):
        cutout_mask(lam_lats, lam_lons, global_lats, global_lons)


def test_cutout_mask_parameter_types():
    """Test that max_distance_km accepts int and float."""
    lam_lat_range = np.linspace(44.0, 46.0, 11)
    lam_lon_range = np.linspace(0.0, 2.0, 11)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    global_lats = np.array([45.0, 46.0])
    global_lons = np.array([0.0, 2.0])

    # Test with int
    mask_int = cutout_mask(lam_lats, lam_lons, global_lats, global_lons, max_distance_km=100)
    assert isinstance(mask_int, np.ndarray)

    # Test with float
    mask_float = cutout_mask(lam_lats, lam_lons, global_lats, global_lons, max_distance_km=100.0)
    assert isinstance(mask_float, np.ndarray)


def test_cutout_mask_large_grid():
    """Test cutout_mask with a larger, more realistic grid."""
    # Create a LAM region (21x21 grid)
    lam_lat_range = np.linspace(40.0, 50.0, 21)
    lam_lon_range = np.linspace(0.0, 10.0, 21)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid (31x31 grid)
    global_lat_range = np.linspace(30.0, 60.0, 31)
    global_lon_range = np.linspace(-10.0, 20.0, 31)
    global_lats, global_lons = np.meshgrid(global_lat_range, global_lon_range)
    global_lats = global_lats.flatten()
    global_lons = global_lons.flatten()

    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        min_distance_km=150.0,
        max_distance_km=300.0,
    )

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (961,)  # 31x31 flattened
    assert mask.dtype == bool
    # Some points should be masked (excluded)
    assert np.any(mask)
    # Some points should not be masked
    assert not np.all(mask)


# ---------------------------------------------------------------------------
# Equivalence of the vectorised cutout_mask with the original per-point
# implementation (kept here as the reference).
# ---------------------------------------------------------------------------


def _reference_cutout_mask(
    lats,
    lons,
    global_lats,
    global_lons,
    cropping_distance=2.0,
    neighbours=5,
    min_distance_km=None,
    max_distance_km=None,
):
    """Original cutout_mask: per-point Python loop over Triangle3D, min/max longitude box."""
    from scipy.spatial import cKDTree

    from anemoi.transform.constants import R_earth_km
    from anemoi.transform.constants import radian
    from anemoi.transform.spatial import Triangle3D
    from anemoi.transform.spatial import _distance_km_to_resolution
    from anemoi.transform.spatial import cropping_mask
    from anemoi.transform.spatial import latlon_to_xyz

    north, south, east, west = np.amax(lats), np.amin(lats), np.amax(lons), np.amin(lons)
    effective_cropping_distance = cropping_distance
    if max_distance_km is not None:
        max_lat = max(abs(north), abs(south))
        L_1_degree_arc_length_km = R_earth_km * np.cos(np.deg2rad(max_lat)) * radian
        effective_cropping_distance = max(cropping_distance, 1.1 * max_distance_km / L_1_degree_arc_length_km)

    mask = cropping_mask(
        global_lats,
        global_lons,
        np.min([90.0, north + effective_cropping_distance]),
        west - effective_cropping_distance,
        np.max([-90.0, south - effective_cropping_distance]),
        east + effective_cropping_distance,
    )
    global_points = np.array(latlon_to_xyz(global_lats[mask], global_lons[mask])).transpose()
    lam_points = np.array(latlon_to_xyz(lats, lons)).transpose()
    min_distance = _distance_km_to_resolution("cutout_mask", min_distance_km, lam_points, global_points)
    distances, indices = cKDTree(lam_points).query(global_points, k=neighbours)
    zero = np.array([0.0, 0.0, 0.0])
    inside_lam = []
    for global_point, distance, index in zip(global_points, distances, indices):
        inside = False
        for j in range(neighbours):
            t = Triangle3D(
                lam_points[index[j]], lam_points[index[(j + 1) % neighbours]], lam_points[index[(j + 2) % neighbours]]
            )
            inside = t.intersect(zero, global_point)
            if inside:
                break
        close = np.min(distance) <= min_distance
        too_far = False
        if max_distance_km is not None:
            too_far = np.min(distance) > (max_distance_km / R_earth_km)
        inside_lam.append(inside or close or too_far)
    too_far_mask = False
    if isinstance(max_distance_km, (int, float)):
        too_far_mask = ~mask.copy()
    mask[mask] = inside_lam
    mask[too_far_mask] = True
    return ~mask


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_distance_km": 50.0},
        {"max_distance_km": 300.0},
        {"min_distance_km": 0.0, "neighbours": 3},
        {"cropping_distance": 0.5, "neighbours": 8},
        {"min_distance_km": 20.0, "max_distance_km": 500.0},
        {"neighbours": 2},
        {"cropping_distance": 0.0},
    ],
)
def test_cutout_mask_matches_reference(kwargs):
    """The vectorised cutout_mask is bit-identical to the original per-point implementation."""
    rng = np.random.default_rng(0)

    # Coarse regular "global" grid
    glat, glon = np.meshgrid(np.arange(-40, 40.01, 2.0), np.arange(-30, 60.01, 2.0), indexing="ij")
    glat, glon = glat.ravel(), glon.ravel()

    # Finer, jittered "LAM" grid over a sub-box so the triangles are irregular
    llat, llon = np.meshgrid(np.arange(-15, 15.01, 0.5), np.arange(0, 30.01, 0.5), indexing="ij")
    llat = llat.ravel() + rng.normal(0, 0.05, llat.size)
    llon = llon.ravel() + rng.normal(0, 0.05, llon.size)

    expected = _reference_cutout_mask(llat, llon, glat, glon, **kwargs)
    actual = cutout_mask(llat, llon, glat, glon, **kwargs)

    assert actual.dtype == np.bool_
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert 0 < actual.sum() < actual.size


@pytest.mark.parametrize("lam_spacing", [0.5, 1.0, 3.0])
def test_cutout_mask_aligned_regular_grids(lam_spacing):
    """Regular grids whose points coincide put global points exactly on triangle edges.

    Those cases are decided by floating point rounding, so the vectorised
    arithmetic must be bit-identical to the reference.
    """
    glat, glon = np.meshgrid(np.arange(-40, 40.01, 0.5), np.arange(-20, 60.01, 0.5), indexing="ij")
    glat, glon = glat.ravel(), glon.ravel()
    llat, llon = np.meshgrid(np.arange(-12, 12.01, lam_spacing), np.arange(9, 33.01, lam_spacing), indexing="ij")
    llat, llon = llat.ravel(), llon.ravel()

    for kwargs in ({}, {"min_distance_km": 30.0}, {"max_distance_km": 400.0}):
        expected = _reference_cutout_mask(llat, llon, glat, glon, **kwargs)
        actual = cutout_mask(llat, llon, glat, glon, **kwargs)
        np.testing.assert_array_equal(actual, expected)


def test_cutout_mask_scattered_lam():
    """Scattered (non-gridded) LAM points give the same answer as the reference."""
    rng = np.random.default_rng(1)
    glat, glon = np.meshgrid(np.arange(-30, 30.01, 1.0), np.arange(-10, 50.01, 1.0), indexing="ij")
    glat, glon = glat.ravel(), glon.ravel()
    llat = rng.uniform(-10, 10, 5000)
    llon = rng.uniform(10, 30, 5000)

    np.testing.assert_array_equal(cutout_mask(llat, llon, glat, glon), _reference_cutout_mask(llat, llon, glat, glon))


def test_longitude_extent():
    from anemoi.transform.spatial import longitude_extent

    assert longitude_extent(np.array([10.0, 20.0, 30.0])) == (10.0, 30.0)
    # Straddling the 0 meridian, [0, 360) convention
    assert longitude_extent(np.array([335.0, 350.0, 0.0, 20.0, 55.0])) == (335.0, 415.0)
    # Straddling the 0 meridian, [-180, 180) convention
    assert longitude_extent(np.array([-25.0, -10.0, 0.0, 20.0, 55.0])) == (335.0, 415.0)
    # Straddling the date line
    assert longitude_extent(np.array([170.0, 180.0, -170.0])) == (170.0, 190.0)
    assert longitude_extent(np.array([42.0])) == (42.0, 42.0)


def test_cutout_mask_lam_across_zero_meridian():
    """A LAM straddling the 0 meridian with longitudes in [0, 360) is handled like any other LAM.

    The original implementation built its cropping box from min/max longitude,
    which spans the whole globe for such a LAM (and made the KD-tree search take
    minutes). Rotating the whole configuration by 180 degrees gives a LAM that
    does not straddle the 0 meridian, for which the reference result is
    trustworthy; the new implementation must reproduce it on the un-rotated
    configuration.
    """
    rng = np.random.default_rng(2)
    glat, glon = np.meshgrid(np.arange(-50, 50.01, 2.0), np.arange(0, 360, 2.0), indexing="ij")
    glat, glon = glat.ravel(), glon.ravel()
    llat, llon = np.meshgrid(np.arange(-15, 15.01, 0.5), np.arange(-25, 35.01, 0.5), indexing="ij")
    llat = llat.ravel() + rng.normal(0, 0.05, llat.size)
    llon = np.mod(llon.ravel() + rng.normal(0, 0.05, llon.size), 360.0)

    expected = _reference_cutout_mask(llat, np.mod(llon + 180, 360), glat, np.mod(glon + 180, 360))
    actual = cutout_mask(llat, llon, glat, glon)
    np.testing.assert_array_equal(actual, expected)
    assert 0 < actual.sum() < actual.size
