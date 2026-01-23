# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd
import tqdm

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("remove_nans")
class RemoveNaNs(Filter):
    """A filter to mask out NaNs.

    All grid points where the first field has NaNs are removed from all fields, and the underlying grid (latitudes and longitudes)
    are updated accordingly. This results in a set of fields on a reduced grid without NaNs.

    Notes
    -----

    This filter will remove NaNs from the input fields by creating a mask based on the first field.
    It assumes that all following fields have NaNs in the same locations as the first field.

    Examples
    --------

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
                param: ...
            - remove_nans: {} # Remove NaNs from all fields based on the first field

    """

    def __init__(self, *, method: str = "mask", check: bool = False, param: str | None = None):
        """Initialize the RemoveNaNs filter.

        Parameters
        ----------
        method : str, optional
            The method to use for removing NaNs, by default "mask".
        check : bool, optional
            Whether to perform a check, by default False.
        param: str, optional
            Param name of the "first" field.
        """

        self.method = method
        self.check = check
        self.param = param

        assert method == "mask", f"Method {method} not implemented"
        assert not check, "Check not implemented"

        self._mask = None
        self._latitudes = None
        self._longitudes = None

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Mask out NaNs in the fields.

        Parameters
        ----------
        fields : ekd.FieldList
            List of fields to be processed.

        Returns
        -------
        ekd.FieldList
            List of fields with NaNs masked out.
        """
        import numpy as np

        if self._mask is None:
            if self.param is None:
                first = fields[0]
            else:
                for first in fields:
                    if first.metadata("param") == self.param:
                        break
                else:
                    raise ValueError(f"{self.param=} not found in\n{fields.ls}")

            data = first.to_numpy(flatten=True)
            self._mask = ~np.isnan(data)

            latitudes, longitudes = first.grid_points()
            self._latitudes = latitudes[self._mask]
            self._longitudes = longitudes[self._mask]

        result = []
        for field in tqdm.tqdm(fields, desc="Remove NaNs"):

            data = field.to_numpy(flatten=True)
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(data[self._mask], template=field),
                    latitudes=self._latitudes,
                    longitudes=self._longitudes,
                )
            )

        return new_fieldlist_from_list(result)
