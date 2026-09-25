# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry


class ConvertUnits(SingleFieldFilter):
    """Rescale a parameter by ``scale``/``offset`` and stamp an explicit units label.

    Unlike the ``convert`` filter (which relies on ``pint`` and therefore cannot emit
    arbitrary unit strings such as ``(0 - 1)``), this filter applies a plain linear
    transform and sets the ``units`` metadata to exactly the string you provide. This
    is needed when the checkpoint expects a units label that is not a valid ``pint``
    unit expression.

    Examples
    --------

    .. code-block:: yaml

      convert_units:
        param: [hcc, mcc, tcc, lcc]
        scale: 0.01
        offset: 0.0
        units: "(0 - 1)"
    """

    required_inputs = ("param", "units")
    optional_inputs = {"scale": 1.0, "offset": 0.0}

    def forward_select(self):
        # param may be a single name or a list of names
        return {"param": self.param}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        values = field.to_numpy() * self.scale + self.offset
        # preserve the field's own param name (important when param is a list)
        param = field.metadata("param", default=field.metadata("shortName", default=None))
        return self.new_field_from_numpy(values, template=field, param=param, units=self.units)


filter_registry.register("convert_units", ConvertUnits)
