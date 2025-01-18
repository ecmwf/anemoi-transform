# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os

from . import Command


class MakeRegridMatrix(Command):
    """Extract the grid from a pair GRIB or NetCDF files extract the MIR interpolation matrix to be used
    by earthkit-regrid."""

    def add_arguments(self, command_parser):
        command_parser.add_argument("--source1", default="file", help="EKD Source type (default: file).")
        command_parser.add_argument("--source2", default="file", help="EKD Source type (default: file).")

        command_parser.add_argument("--mir", default=os.environ.get("MIR_COMMAND", "mir"), help="MIR command")

        command_parser.add_argument("input1", help="Input file (GRIB or NetCDF).")
        command_parser.add_argument("input2", help="Input file (GRIB or NetCDF).")

        command_parser.add_argument("output", help="Output NPZ file.")

        command_parser.add_argument("kwargs", nargs="*", help="MIR arguments.")

    def run(self, args):
        from earthkit.data import from_source
        from earthkit.regrid.utils.mir import mir_make_matrix

        ds1 = from_source(args.source1, args.input1)
        lat1, lon1 = ds1[0].grid_points()

        ds2 = from_source(args.source2, args.input2)
        lat2, lon2 = ds2[0].grid_points()

        kwargs = {}
        for arg in args.kwargs:
            key, value = arg.split("=")
            kwargs[key] = value

        mir_make_matrix(args.output, lat1, lon1, lat2, lon2, mir=args.mir, **kwargs)


command = MakeRegridMatrix
