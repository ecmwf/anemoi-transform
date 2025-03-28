# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict

import earthkit.data as ekd

from anemoi.transform.source import Source
from anemoi.transform.sources import source_registry


@source_registry.register("mars")
class Mars(Source):
    """A demo source."""

    def __init__(self, **request: Any) -> None:
        """Initialize the Mars source.

        Parameters
        ----------

        **request : Any
            Keyword arguments for the MARS request.
        """

    def forward(self, data: Dict[str, Any]) -> ekd.Source:
        """Fetch data from MARS.

        Parameters
        ----------
        data : Dict[str, Any]
            The request parameters for fetching data from MARS.

        Returns
        -------
        ekd.Source
            The data fetched from MARS.
        """
        return ekd.from_source("mars", **data)

    def __ror__(self, data: Dict[str, Any]) -> Source:
        """Enable the use of the pipe operator with this source.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data to be processed.

        Returns
        -------
        Source
            An Input source that processes the data.
        """
        this = self

        class Input(Source):
            def __init__(self, *, data: Dict[str, Any]) -> None:
                """Initialize the Input source.

                Parameters
                ----------
                data : Dict[str, Any]
                    The input data to be processed.
                """
                self.data: Dict[str, Any] = data

            def forward(self, data: Any) -> ekd.Source:
                """Fetch data from MARS using the stored request parameters.

                Parameters
                ----------
                data : Any
                    The input data to be processed.

                Returns
                -------
                ekd.Source
                    The data fetched from MARS.
                """
                return this.forward(self.data)

        return Input(data)
