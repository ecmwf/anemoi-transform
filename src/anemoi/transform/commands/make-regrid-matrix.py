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

LOG = logging.getLogger(__name__)


def get_lat_lon(ds):
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

    raise ValueError("No grid points found in the dataset.")


def check(input_file, latitudes, longitudes):
    LOG.info(f"Checking for duplicate lat/lon pairs in {input_file}...")
    seen = set()
    for lat, lon in zip(latitudes, longitudes):
        if (lat, lon) in seen:
            raise ValueError(f"Duplicate latitude/longitude pair found in {input_file}: ({lat}, {lon})")
        seen.add((lat, lon))


def round_lat_lon(latitudes, longitudes, rounding):
    import numpy as np

    # 1 deg = 111.32 km
    D = 111.32 * 1000  # in meters
    for n in range(rounding):
        D /= 10
    LOG.info(f"Rounding latitudes and longitudes to {rounding} decimal places ({D} m).")
    return np.round(latitudes, rounding), np.round(longitudes, rounding)


def make_mir_matrix(lat1, lon1, lat2, lon2, output=None, mir="mir", **kwargs):

    import numpy as np
    from earthkit.regrid.utils.mir import mir_make_matrix

    sparse_array = mir_make_matrix(lat1, lon1, lat2, lon2, output=None, mir=mir, **kwargs)

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


def make_global_on_lam_mask(lat1, lon1, lat2, lon2, output=None, **kwargs):
    import numpy as np

    from anemoi.transform.spatial import global_on_lam_mask

    mask = global_on_lam_mask(lat1, lon1, lat2, lon2, **kwargs)
    if output is not None:
        np.savez(output, mask)


class MakeRegridMatrix(Command):
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
        command_parser.add_argument("--source1", default="file", help="EKD Source type (default: file).")
        command_parser.add_argument("--source2", default="file", help="EKD Source type (default: file).")

        command_parser.add_argument("--mir", default=os.environ.get("MIR_COMMAND", "mir"), help="MIR command")

        command_parser.add_argument(
            "--rounding", type=int, help="Round latitudes and longitudes to this precision (default: None)."
        )
        command_parser.add_argument(
            "--check", action="store_true", help="Check for duplicate lat/lon pairs in the input files."
        )
        command_parser.add_argument("input1", help="Input file (GRIB or NetCDF).")
        command_parser.add_argument("input2", help="Input file (GRIB or NetCDF).")

        command_parser.add_argument("output", help="Output NPZ file.")

        command_parser.add_argument("kwargs", nargs="*", help="MIR arguments.")

    def run(self, args: argparse.Namespace) -> None:
        """Run the command with the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to run the command with.
        """
        import numpy as np
        from earthkit.data import from_source

        _, ext1 = os.path.splitext(args.input1)
        if ext1 in (".npz", ".npy"):
            ds1 = np.load(args.input1)
            lat1 = ds1["latitudes"]
            lon1 = ds1["longitudes"]
        else:
            ds1 = from_source(args.source1, args.input1)
            lat1, lon1 = get_lat_lon(ds1)

        _, ext2 = os.path.splitext(args.input2)
        if ext2 in (".npz", ".npy"):
            ds2 = np.load(args.input2)
            lat2 = ds2["latitudes"]
            lon2 = ds2["longitudes"]
        else:
            ds2 = from_source(args.source2, args.input2)
            lat2, lon2 = get_lat_lon(ds2)

        kwargs = {}
        for arg in args.kwargs:
            key, value = arg.split("=")
            kwargs[key] = value

        if args.rounding is not None:
            lat1, lon1 = round_lat_lon(lat1, lon1, args.rounding)
            lat2, lon2 = round_lat_lon(lat2, lon2, args.rounding)

        if args.check:
            check(args.input1, lat1, lon1)
            check(args.input2, lat2, lon2)

        LOG.info(f"Creating MIR interpolation matrix from {args.input1} to {args.input2}...")
        sparse_array = mir_make_matrix(lat1, lon1, lat2, lon2, output=None, mir=args.mir, **kwargs)

        LOG.info("MIR interpolation matrix created successfully.")
        np.savez(
            args.output,
            matrix_data=sparse_array.data,
            matrix_indices=sparse_array.indices,
            matrix_indptr=sparse_array.indptr,
            matrix_shape=sparse_array.shape,
            in_latitudes=lat1,
            in_longitudes=lon1,
            out_latitudes=lat2,
            out_longitudes=lon2,
        )


command = MakeRegridMatrix
