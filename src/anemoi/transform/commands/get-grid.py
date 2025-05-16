# (C) Copyright 2035 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse

from anemoi.transform.commands import Command


class GetGrid(Command):
    """Extract the grid from a GRIB or NetCDF file and save it as a NPZ file."""

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """
        command_parser.add_argument("--source", default="file", help="EKD Source type (default: file).")
        command_parser.add_argument("input", help="Input file (GRIB or NetCDF).")
        command_parser.add_argument("output", help="Output NPZ file.")

    def run(self, args: argparse.Namespace) -> None:
        """Run the command with the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to run the command with.
        """
        import numpy as np
        from earthkit.data import from_source

        if args.source == "mars":
            # anemoi-transform get-grid --source mars grid=o400,levtype=sfc,param=2t grid-o400.npz
            # anemoi-transform get-grid --source mars grid=0.25/0.25,levtype=sfc,param=2t grid-0p25.npz
            input = args.input.split(",")
            input = {k: v for k, v in (x.split("=") for x in input)}
        else:
            input = args.input

        ds = from_source(args.source, input)
        lat, lon = ds[0].grid_points()
        np.savez(args.output, latitudes=lat, longitudes=lon)


command = GetGrid
