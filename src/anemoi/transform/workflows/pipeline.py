# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import List

import earthkit.data as ekd

from anemoi.transform.workflow import Workflow
from anemoi.transform.workflows import workflow_registry


@workflow_registry.register("pipeline")
class Pipeline(Workflow):
    """A simple pipeline of filters.

    Parameters
    ----------

    filters : List[Any]
        A list of filter objects that have `forward` and `backward` methods.
    """

    def __init__(self, *, filters: List[Any]) -> None:

        self.filters: List[Any] = filters

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply the filters in sequence to the data.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be processed by the filters.

        Returns
        -------
        ekd.FieldList
            The processed data after applying all filters.
        """
        for filter in self.filters:
            data = filter.forward(data)
        return data

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply the filters in reverse sequence to the data.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be processed by the filters in reverse order.

        Returns
        -------
        ekd.FieldList
            The processed data after applying all filters in reverse order.
        """
        for filter in reversed(self.filters):
            data = filter.backward(data)
        return data
