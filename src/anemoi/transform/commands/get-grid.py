# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import Command


class GetGrid(Command):
    """Extract the grid from a GRIB or NetCDF file and save it as a NPZ file."""

    def add_arguments(self, command_parser):
        command_parser.add_argument("--source", default="file", help="EKD Source type (default: file).")
        command_parser.add_argument("input", help="Input file (GRIB or NetCDF).")
        command_parser.add_argument("output", help="Output NPZ file.")

    def run(self, args):
        import numpy as np
        from earthkit.data import from_source

        ds = from_source(args.source, args.input)
        lat, lon = ds[0].grid_points()
        np.savez(args.output, latitudes=lat, longitudes=lon)


command = GetGrid
