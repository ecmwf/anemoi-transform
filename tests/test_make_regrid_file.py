# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``-z/--compress`` option on ``make-regrid-file global-on-lam-mask``
(#200), covering both the numpy-facing static method and the argparse wiring.

The command module lives at ``commands/make-regrid-file.py`` -- a hyphenated
filename that can't be named in an ``import`` statement -- so it's fetched via
``importlib``, after ``anemoi.transform.commands`` has registered it (which is
also how the package itself loads it, see ``commands/__init__.py``).
"""

import argparse
import importlib
import os

import numpy as np

import anemoi.transform.commands  # noqa: F401  (registers COMMANDS, see module docstring)

make_regrid_file = importlib.import_module("anemoi.transform.commands.make-regrid-file")
MakeGlobalOnLamMask = make_regrid_file.MakeGlobalOnLamMask


def _make_grids():
    """A small LAM region nested inside a larger global grid, as in test_spatial.py."""
    lam_lat_range = np.linspace(44.0, 46.0, 11)
    lam_lon_range = np.linspace(0.0, 2.0, 11)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    global_lat_range = np.linspace(30.0, 60.0, 101)
    global_lon_range = np.linspace(-10.0, 20.0, 101)
    global_lats, global_lons = np.meshgrid(global_lat_range, global_lon_range)
    global_lats = global_lats.flatten()
    global_lons = global_lons.flatten()

    return lam_lats, lam_lons, global_lats, global_lons


def test_compress_flag_produces_identical_mask(tmp_path):
    """Regression (#200): compress=True must not change the resulting mask,
    only how it's stored on disk.
    """
    lam_lats, lam_lons, global_lats, global_lons = _make_grids()

    plain_path = tmp_path / "plain.npz"
    compressed_path = tmp_path / "compressed.npz"

    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(plain_path),
    )
    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(compressed_path),
        compress=True,
    )

    plain_mask = np.load(plain_path)["mask"]
    compressed_mask = np.load(compressed_path)["mask"]
    assert np.array_equal(plain_mask, compressed_mask)


def test_compress_flag_actually_shrinks_file(tmp_path):
    """A boolean mask over a 101x101 grid is highly redundant, so a real
    compressed write should end up smaller -- guards against a no-op
    implementation that pops the kwarg but still calls plain np.savez.
    """
    lam_lats, lam_lons, global_lats, global_lons = _make_grids()

    plain_path = tmp_path / "plain.npz"
    compressed_path = tmp_path / "compressed.npz"

    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(plain_path),
    )
    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(compressed_path),
        compress=True,
    )

    assert os.path.getsize(compressed_path) < os.path.getsize(plain_path)


def test_compress_default_is_false_and_backwards_compatible(tmp_path):
    """Regression: existing callers that never pass `compress` (e.g. any
    external script built against the old signature) must keep working
    exactly as before.
    """
    lam_lats, lam_lons, global_lats, global_lons = _make_grids()
    output = tmp_path / "default.npz"

    # Must not raise -- omitting `compress` entirely is the pre-#200 call shape.
    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(output),
    )
    saved = np.load(output)["mask"]
    assert isinstance(saved, np.ndarray)
    assert saved.size > 0


def test_compress_does_not_leak_into_global_on_lam_mask_kwargs(tmp_path):
    """Regression: `compress` must be popped from kwargs before they reach
    `global_on_lam_mask`, which has no such parameter -- passing it straight
    through would raise TypeError. Also checks a genuine kwarg (distance_km)
    still passes through untouched alongside it.
    """
    lam_lats, lam_lons, global_lats, global_lons = _make_grids()
    output = tmp_path / "with_distance.npz"

    MakeGlobalOnLamMask.make_global_on_lam_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        output=str(output),
        compress=True,
        distance_km=200.0,
    )
    assert output.exists()


def test_compress_cli_flag_parses_to_true():
    parser = argparse.ArgumentParser()
    MakeGlobalOnLamMask().add_arguments(parser)
    args = parser.parse_args(
        [
            "--lam-grid",
            "lam.grib",
            "--global-grid",
            "global.grib",
            "--output",
            "out.npz",
            "-z",
        ]
    )
    assert args.compress is True


def test_compress_cli_flag_defaults_to_false():
    parser = argparse.ArgumentParser()
    MakeGlobalOnLamMask().add_arguments(parser)
    args = parser.parse_args(
        [
            "--lam-grid",
            "lam.grib",
            "--global-grid",
            "global.grib",
            "--output",
            "out.npz",
        ]
    )
    assert args.compress is False


def test_run_forwards_compress_to_make_global_on_lam_mask(monkeypatch, tmp_path):
    """Regression: `run()` must thread args.compress through, not just
    `add_arguments()` -- a flag that parses fine but is silently dropped
    before reaching the static method would pass the two tests above while
    the CLI still never compressed anything.
    """
    captured = {}

    def fake_make_global_on_lam_mask(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        MakeGlobalOnLamMask,
        "make_global_on_lam_mask",
        staticmethod(fake_make_global_on_lam_mask),
    )
    monkeypatch.setattr(
        make_regrid_file,
        "_path_to_lat_lon",
        lambda path: (np.array([0.0]), np.array([0.0])),
    )

    args = argparse.Namespace(
        lam_grid="lam.grib",
        global_grid="global.grib",
        output=str(tmp_path / "out.npz"),
        plot=None,
        distance_km=None,
        compress=True,
    )
    MakeGlobalOnLamMask().run(args)

    assert captured.get("compress") is True
