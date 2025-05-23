# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Dict
from typing import Iterator
from typing import Optional

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

LOG = logging.getLogger(__name__)


@filter_registry.register("timeseries")
class Timeseries(MatchingFieldsFilter):
    """A source to add a timeseries depending on time but not on location.

    Parameters
    ----------
    netcdf : dict, optional
        Dictionary containing the path to the netCDF file, by default None.
    template_param : str, optional
        Template parameter name, by default "2t".
    """

    @matching(
        select="param",
        forward="template_param",
    )
    def __init__(
        self,
        *,
        netcdf: Optional[Dict[str, str]] = None,
        template_param: str = "2t",
    ) -> None:

        if netcdf:
            import xarray as xr

            self.ds = xr.open_dataset(netcdf["path"])  # .to_dataframe()
        LOG.warning("Using the timeseries filter will be deprecated in the future. Please do not rely on it.")

        self.template_param = template_param

    def forward_transform(self, template: ekd.Field) -> Iterator[ekd.Field]:
        """Convert snow depth and snow density to snow cover.

        Parameters
        ----------
        template : ekd.Field
            Template field to transform.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields.
        """
        dt = template.metadata("valid_datetime")
        template_array = template.to_numpy()

        sel = self.ds.sel(time=dt)

        for name in self.ds.data_vars:
            value = sel[name].values
            data = np.full_like(template_array, value)
            yield self.new_field_from_numpy(data, template=template, param=name)
