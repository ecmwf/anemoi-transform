# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import numpy as np
import tqdm

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters.fields import filter_registry
from anemoi.transform.grids.latlon import LatLonGrid

LOG = logging.getLogger(__name__)


@filter_registry.register("canonical_order")
class CanonicalOrder(Filter):
    """Reorder field data to the canonical grid ordering.

    Canonical order is latitude descending (N->S), longitude ascending
    in [0, 360), row-major. Fields already in canonical order pass
    through unchanged. The permutation is computed once from the first
    field and reused for all subsequent fields, which are validated to
    be on the same ordering.

    Examples
    --------
    .. code-block:: yaml

        input:
          pipe:
            - source: ...
            - canonical_order: {}
    """

    def __init__(self, *, check: bool = True):
        self.check = check
        self._permutation = None
        self._latitudes = None
        self._longitudes = None
        self._reference_grid = None
        self._identity = False

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        if self._permutation is None:
            first = fields[0]
            grid = LatLonGrid.from_field(first)
            self._reference_grid = grid
            self._identity = grid.is_canonical()
            if self._identity:
                LOG.info("Fields already in canonical order, passing through")
                self._permutation = np.arange(grid.size)
                self._latitudes, self._longitudes = grid.latlon()
            else:
                self._permutation = grid.canonical_permutation()
                canonical = grid.canonical()
                self._latitudes, self._longitudes = canonical.latlon()

        if self._identity and not self.check:
            return fields

        result = []
        for field in tqdm.tqdm(fields, desc="Canonical order"):
            if self.check and not LatLonGrid.from_field(field).same_ordering_as(self._reference_grid):
                raise ValueError("Field is not on the same grid ordering as the first field")
            if self._identity:
                result.append(field)
                continue
            data = field.to_numpy(flatten=True)
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(data[self._permutation], template=field),
                    latitudes=self._latitudes,
                    longitudes=self._longitudes,
                )
            )
        return new_fieldlist_from_list(result)
