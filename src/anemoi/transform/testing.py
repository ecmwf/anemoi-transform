# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import earthkit.data as ekd
from anemoi.utils.testing import get_test_data

from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.source import Source


def fieldlist_fixture(name: str = "2t-sp.grib") -> Any:
    """Fixture to create a fieldlist for testing.

    Parameters
    ----------
    name : str, optional
        The name of the fieldlist to create, by default "2t-sp.grib".

    Returns
    -------
    Any
        The created fieldlist.
    """
    return ekd.from_source("file", get_test_data(f"anemoi-filters/{name}"))


class SelectFieldSource(Source):
    def __init__(self, fields, params=None):
        self._fields = fields
        self.params = params

    def forward(self, *args, **kwargs):
        fields = []
        for f in self._fields:
            if self.params and f.metadata("param") in self.params:
                fields.append(f)
        return new_fieldlist_from_list(fields)
