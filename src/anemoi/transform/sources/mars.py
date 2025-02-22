# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

import earthkit.data as ekd

from ..source import Source
from . import source_registry


@source_registry.register("mars")
class Mars(Source):
    """A demo source."""

    def __init__(self, **request: Any) -> None:
        pass

    def forward(self, data: Any) -> Any:
        return ekd.from_source("mars", **data)

    def __ror__(self, data: Any) -> Any:

        this = self

        class Input(Source):
            def __init__(self, data: Any) -> None:
                self.data = data

            def forward(self, data: Any) -> Any:
                return this.forward(self.data)

        return Input(data)
