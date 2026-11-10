# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib.util
import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import anemoi.transform.commands as commands  # noqa: E402

ZARR_3 = zarr.__version__ >= "3"
ZARR_FORMATS = [2, 3] if ZARR_3 else [2]


def _load_module():
    # the module file name has hyphens, so it cannot be imported with a plain import statement
    path = os.path.join(os.path.dirname(commands.__file__), "make-regrid-file.py")
    spec = importlib.util.spec_from_file_location("make_regrid_file", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _open_group(path, zarr_format):
    if ZARR_3:
        return zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    return zarr.open_group(str(path), mode="w")


def _write_array(group, name, data):
    if ZARR_3:
        array = group.create_array(name, data=data)
    else:
        array = group.create_dataset(name, data=data)
    # anemoi-datasets stores the dimension names in this attribute only (no ``dimension_names`` in format 3)
    array.attrs["_ARRAY_DIMENSIONS"] = ["cell"]


@pytest.mark.parametrize("zarr_format", ZARR_FORMATS)
def test_path_to_lat_lon_anemoi_zarr(tmp_path, zarr_format):
    """The latitudes and longitudes of an anemoi dataset are read with zarr, whatever the zarr format."""
    # no value equals the default fill value (0), which xarray would mask as NaN
    latitudes = np.linspace(-80.0, 80.0, 12)
    longitudes = np.linspace(5.0, 335.0, 12)
    group = _open_group(tmp_path / "anemoi.zarr", zarr_format)
    _write_array(group, "latitudes", latitudes)
    _write_array(group, "longitudes", longitudes)

    lat, lon = _load_module()._path_to_lat_lon(str(tmp_path / "anemoi.zarr"))

    np.testing.assert_array_equal(lat, latitudes)
    np.testing.assert_array_equal(lon, longitudes)


@pytest.mark.parametrize("kind", ["group_without_arrays", "empty_directory", "missing_path"])
def test_path_to_lat_lon_not_an_anemoi_zarr_uses_fallback(tmp_path, monkeypatch, kind):
    """A path that is not an anemoi dataset goes to the earthkit-data fallback, as before."""
    path = tmp_path / "other.zarr"
    if kind == "group_without_arrays":
        _write_array(_open_group(path, 2), "x", np.zeros(3))
    elif kind == "empty_directory":
        path.mkdir()
    module = _load_module()
    expected = (np.zeros(1), np.ones(1))
    monkeypatch.setattr(module, "_ds_to_lat_lon", lambda p: expected)

    assert module._path_to_lat_lon(str(path)) is expected
