# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import earthkit.data as ekd

from anemoi.transform.filters import Filter
from anemoi.transform.filters import register_filter


class Mars(Filter):
    """A demo source"""

    def __init__(self, **request):
        self.request = request

    def forward(self, data):
        assert data is None

        return ekd.from_source("mars", **self.request)

    def backward(self, data):
        raise NotImplementedError("Cannot archive data in MARS that way")


register_filter("mars", Mars)
