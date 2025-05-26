# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("clip")
class Clipper(MatchingFieldsFilter):
    """Clip the values of a single field to a specified range [minimum, maximum].
    Values below `minimum` will be set to `minimum`, and values above `maximum` will be set to `maximum`.
    At least one of `minimum` or `maximum` must be specified.

    Parameters
    ----------
    param : str
        Name of the field to clip.
    minimum : float, optional
        Minimum allowed value. Values below this will be set to minimum.
    maximum : float, optional
        Maximum allowed value. Values above this will be set to maximum.
    """

    @matching(select="param", forward=("param",))
    def __init__(
        self,
        *,
        param: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ):
        if minimum is None and maximum is None:
            raise ValueError("At least one value for minimum or maximum must be specified.")
        self.param = param
        self.minimum = minimum
        self.maximum = maximum

    def forward_transform(self, param: ekd.Field) -> Iterator[ekd.Field]:
        data = param.to_numpy()
        clipped = np.clip(data, self.minimum, self.maximum)
        yield self.new_field_from_numpy(clipped, template=param, param=self.param)
