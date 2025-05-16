# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import os

from anemoi.transform.commands import Command


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
        from earthkit.regrid.utils.mir import mir_make_matrix

        _, ext1 = os.path.splitext(args.input1)
        if ext1 in (".npz", ".npy"):
            ds1 = np.load(args.input1)
            lat1 = ds1["latitudes"]
            lon1 = ds1["longitudes"]
        else:
            ds1 = from_source(args.source1, args.input1)
            lat1, lon1 = ds1[0].grid_points()

        _, ext2 = os.path.splitext(args.input2)
        if ext2 in (".npz", ".npy"):
            ds2 = np.load(args.input2)
            lat2 = ds2["latitudes"]
            lon2 = ds2["longitudes"]
        else:
            ds2 = from_source(args.source2, args.input2)
            lat2, lon2 = ds2[0].grid_points()

        kwargs = {}
        for arg in args.kwargs:
            key, value = arg.split("=")
            kwargs[key] = value

        sparse_array = mir_make_matrix(lat1, lon1, lat2, lon2, output=None, mir=args.mir, **kwargs)

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
