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


class Filters(Command):
    """Command to inspect available filters"""

    def add_arguments(self, command_parser: argparse.ArgumentParser) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)
        subparsers.add_parser(
            "list",
            help="List avaialble filters",
            description="List available filters",
        )

    def run(self, args: argparse.Namespace) -> None:
        """Run the command with the provided arguments.

        Parameters
        ----------
        args : argparse.Namespace
            The arguments to run the command with.
        """
        if args.subcommand == "list":
            from anemoi.transform.filters import filter_registry

            filters = filter_registry.registered
            print("Available Filters:\n" + "-" * 18)
            for f in sorted(filters):
                print(f"- {f}")


command = Filters
