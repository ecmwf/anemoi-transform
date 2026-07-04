# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.transform.naming import naming_registry
from anemoi.transform.naming.template import TemplateNaming


@naming_registry.register("param_levelist")
class ParamLevelNaming(TemplateNaming):
    """Name fields by variable name and level (e.g. ``"2t"``, ``"t_850"``).

    This is the default naming scheme: the variable name, an underscore,
    then the level; the level part is omitted for surface fields.
    """

    def __init__(self) -> None:
        super().__init__("{param}_{levelist}")
