# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator, Optional
import numpy as np
import earthkit.data as ekd

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("clipper")
class Clipper(MatchingFieldsFilter):
    """Clip the values of a single field to a specified range [minimum_valueue, maximum_valueue].

    Parameters
    ----------
    param : str
        Name of the field to clip.
    minimum_valueue : float, optional
        Minimum allowed value. Values below this will be set to minimum_valueue.
    maximum_valueue : float, optional
        Maximum allowed value. Values above this will be set to maximum_valueue.
    """

    @matching(select="param", forward=("param",))
    def __init__(
        self,
        *,
        param: str,
        minimum_valueue: Optional[float] = None,
        maximum_valueue: Optional[float] = None,
    ):
        if minimum_valueue is None and maximum_valueue is None:
            raise ValueError("At least one of minimum_valueue or maximum_valueue must be specified.")
        self.param = param
        self.minimum_valueue = minimum_valueue
        self.maximum_valueue = maximum_valueue

    def forward_transform(self, param: ekd.Field) -> Iterator[ekd.Field]:
        data = param.to_numpy()
        clipped = np.clip(data, self.minimum_valueue, self.maximum_valueue)
        yield self.new_field_from_numpy(clipped, template=param, param=self.param)
