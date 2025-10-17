# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import os

from anemoi.transform.commands import Command
from anemoi.transform.constants import L_1_degree_earth_arc_length_km as L_1d_km

LOG = logging.getLogger(__name__)


def _ds_to_lat_lon(ds):
    try:
        return ds[0].grid_points()
    except TypeError:
        # This is a workaround for datasets that do not have data variables,
        # but have latitude and longitude coordinates.
        import xarray as xr

        ds = xr.open_dataset(ds.path)
        lat = ds["latitude"].values.flatten()
        lon = ds["longitude"].values.flatten()
        return lat, lon


def _path_to_lat_lon(path):
    """Extract latitudes and longitudes from a file path."""
    import earthkit.data as ekd
    import numpy as np

    rf path.endswith(".npz"):
        data = np.load(path)
        return data["latitudes"], data["longitudes"]
    if path.endswith(".zarr"):
        from anemoi.datasets import open_dataset
        dataset = open_dataset(path)
        return dataset.latitudes, dataset.longitudes
    ds = ekd.from_source("file", path)
    return _ds_to_lat_lon(ds)


def check_duplicate_latlons(input_file, latitudes, longitudes):
    LOG.info(f"Checking for duplicate lat/lon pairs in {input_file}...")
    seen = set()
    for lat, lon in zip(latitudes, longitudes):
        if (lat, lon) in seen:
            raise ValueError(f"Duplicate latitude/longitude pair found in {input_file}: ({lat}, {lon})")
        seen.add((lat, lon))


def round_lat_lon(latitudes, longitudes, rounding):
    import numpy as np
    LOG.info(f"Rounding latitudes and longitudes to {rounding} decimal places ({L_1d_km / ( 10 ) ** rounding} m).")
    return np.round(latitudes, rounding), np.round(longitudes, rounding)


class MakeMIRMatrix:
    """Extract the grid from a pair GRIB or NetCDF files extract the MIR interpolation matrix to be used
    by earthkit-regrid.
    """

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """
        command_parser.add_argument("--source-grid", help="Input file (GRIB or NetCDF).", required=True)
        command_parser.add_argument("--target-grid", help="Input file (GRIB or NetCDF).", required=True)

        command_parser.add_argument(
            "--mir", default=os.environ.get("MIR_COMMAND", "mir"), help="MIR command (default: 'mir')."
        )
        command_parser.add_argument(
            "--rounding",
            type=int,
            help="Round latitudes and longitudes to this precision (default: None).",
        )
        command_parser.add_argument("--check", action="store_true", help="Check for duplicate lat/lon pairs.")
        command_parser.add_argument(
            "--mir_args", 
            nargs="*", help="MIR arguments. Usage: --mir_args arg1=val1 arg2=val2 ...", 
            type=lambda kv: kv.split("="))
        command_parser.add_argument(
            "--output",
            type=str,
            help="Output NPZ file. The name will be used to determine the type of regrid file to create.",
            required=True,
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run the command with the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to run the command with.
        """
        mir_kwargs = dict(args.mir_kwargs) if args.mir_kwargs is not None else {}
        source_lat, source_lon = _path_to_lat_lon(args.source_grid)
        target_lat, target_lon = _path_to_lat_lon(args.target_grid)

        if args.rounding is not None:
            source_lat, source_lon = round_lat_lon(source_lat, source_lon, args.rounding)
            target_lat, target_lon = round_lat_lon(target_lat, target_lon, args.rounding)

        if args.check:
            check_duplicate_latlons(args.source_grid, source_lat, source_lon)
            check_duplicate_latlons(args.target_grid, target_lat, target_lon)

        MakeMIRMatrix.make_mir_matrix(source_lat, source_lon, target_lat, target_lon, output=args.output, mir=args.mir, **mir_kwargs)
    
    @staticmethod
    def make_mir_matrix(lat1, lon1, lat2, lon2, output=None, mir="mir", **mir_kwargs):

        import numpy as np
        from earthkit.regrid.utils.mir import mir_make_matrix

        sparse_array = mir_make_matrix(lat1, lon1, lat2, lon2, output=None, mir=mir, **mir_kwargs)

        np.savez(
            output,
            matrix_data=sparse_array.data,
            matrix_indices=sparse_array.indices,
            matrix_indptr=sparse_array.indptr,
            matrix_shape=sparse_array.shape,
            in_latitudes=lat1,
            in_longitudes=lon1,
            out_latitudes=lat2,
            out_longitudes=lon2,
        )

class MakeGlobalOnLamMask:

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """
        command_parser.add_argument("--lam-grid", help="Input file (GRIB or NetCDF).", required=True)
        command_parser.add_argument("--global-grid", help="Input file (GRIB or NetCDF).", required=True)
        command_parser.add_argument(
            "--distance-km",
            type=float,
            default=None,
            help="Distance in kilometers to consider for the mask. If None, use the spacing of the global grid.",
        )
        command_parser.add_argument(
            "--output",
            type=str,
            help="Output NPZ file. The name will be used to determine the type of regrid file to create.",
            required=True,
        )
        command_parser.add_argument(
            "--plot",
            type=str,
            help="A path in which to plot the mask.",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run the command with the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to run the command with.
        """

        lam_lat, lam_lon = _path_to_lat_lon(args.lam_grid)
        global_lat, global_lon = _path_to_lat_lon(args.global_grid)

        MakeGlobalOnLamMask.make_global_on_lam_mask(
            lam_lat, lam_lon, global_lat, global_lon, output=args.output, distance_km=args.distance_km, plot=args.plot
        )

    @staticmethod
    def make_global_on_lam_mask(lam_lat, lam_lon, global_lat, global_lon, output, **kwargs):
        import numpy as np

        from anemoi.transform.spatial import global_on_lam_mask

        mask = global_on_lam_mask(lam_lat, lam_lon, global_lat, global_lon, **kwargs)
        np.savez(output, mask=mask)


OPTIONS = {
    "mir-matrix": MakeMIRMatrix,
    "global-on-lam-mask": MakeGlobalOnLamMask,
}


class MakeRegridFile(Command):
    """Extract the grid from a pair GRIB or NetCDF files extract the MIR interpolation matrix to be used
    by earthkit-regrid.
    """

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """

        subparsers = command_parser.add_subparsers(dest="type", required=True)

        for k, v in OPTIONS.items():
            subparser = subparsers.add_parser(k, help=v.__doc__)
            v().add_arguments(subparser)

    def run(self, args: argparse.Namespace) -> None:
        OPTIONS[args.type]().run(args)


command = MakeRegridFile
