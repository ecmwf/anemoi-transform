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


@naming_registry.register("param")
class ParamNaming(TemplateNaming):
    """Name fields by their variable name alone (e.g. ``"2t"``, ``"t"``)."""

    def __init__(self) -> None:
        super().__init__("{param}")
