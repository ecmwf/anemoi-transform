# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from anemoi.transform.filters import Filter
from anemoi.transform.filters import register_filter


class Pipeline(Filter):
    """A simple pipeline of filters"""

    def __init__(self, filters):
        self.filters = filters

    def forward(self, data):
        for filter in self.filters:
            data = filter.forward(data)
        return data

    def backward(self, data):
        for filter in reversed(self.filters):
            data = filter.backward(data)
        return data

    def __iter__(self):
        return iter(self(None))


register_filter("pipeline", Pipeline)
