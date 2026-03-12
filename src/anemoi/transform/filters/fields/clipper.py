# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import numpy as np

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


@filter_registry.register("clip")
class Clipper(SingleFieldFilter):
    """Clip the values of a single field to a specified range [minimum, maximum].

    Values below `minimum` will be set to `minimum`, and values above `maximum` will be set to `maximum`.
    At least one of `minimum` or `maximum` must be specified.

    Clipping is defined as:

    If ``maximum`` is not specified:

    .. math::

        \\operatorname{clip}(x,minimum) = \\max(x,minimum).

    If ``minimum`` is not specified:

    .. math::

        \\operatorname{clip}(x,maximum) = \\min(x,maximum).

    If both ``minimum`` and ``maximum`` are specified:

    .. math::

        \\operatorname{clip}(x,minimum,maximum) = \\min(\\max(x,minimum),maximum).

    Examples
    --------
    Clip the total precipitation rate (`tp`) to [0, ♾️):

    .. code-block:: yaml

        clip:
            param: tp
            minimum: 0.0

    """

    required_inputs = ("param",)
    optional_inputs = {"minimum": None, "maximum": None}

    def prepare_filter(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("At least one value for minimum or maximum must be specified.")

    def forward_select(self):
        return {"param": self.param}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        data = field.to_numpy()
        clipped = np.clip(data, self.minimum, self.maximum)
        return self.new_field_from_numpy(clipped, template=field, param=self.param)
