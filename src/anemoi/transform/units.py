# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


class Units:
    def __init__(self, units: str) -> None:
        self.units = units

    def __str__(self) -> str:
        return self.units

    def __repr__(self) -> str:
        return self.units

    def __eq__(self, value):
        if isinstance(value, Units):
            return self.units == value.units
        elif isinstance(value, str):
            return self.units == value
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.units)
