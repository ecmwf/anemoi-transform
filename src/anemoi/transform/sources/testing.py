# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
from anemoi.utils.testing import get_test_data

from anemoi.transform.source import Source
from anemoi.transform.sources import source_registry


@source_registry.register("testing")
class Testing(Source):
    """A demo source."""

    def __init__(self, name) -> None:
        self.path = get_test_data(name)

    def forward(self, *args, **kwargs) -> ekd.Source:
        return ekd.from_source("file", self.path)
