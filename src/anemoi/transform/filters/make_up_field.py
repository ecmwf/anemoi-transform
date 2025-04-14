# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Iterator

import earthkit.data as ekd

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class MakeUpField(MatchingFieldsFilter):
    """A filter to simply copy the data of a template field and
    and back.
    """

    @matching(
        select="param",
        forward=("template"),
    )
    def __init__(
        self,
        *,
        template="t",
        makeupname="skt",
    ):

        self.template = template
        self.makeupname = makeupname

    def forward_transform(
        self,
        template: ekd.Field,
    ) -> Iterator[ekd.Field]:

        yield self.new_field_from_numpy(template.to_numpy(), template=template, param=self.makeupname)
        yield template


filter_registry.register("makeup-field", MakeUpField)
