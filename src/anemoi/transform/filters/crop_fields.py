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


from anemoi.datasets.grids import cropping_mask
from numpy.typing import NDArray

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("crop")
class CropWithMask(Filter):
    """A filter to crop fields inside a simple lat/lon mask."""

    def __init__(self, *, area: dict = {'north': 90, 'west':0, 'south':-90, 'east':360}):
        """Initialize the CropWithMask filter.

        Parameters
        ----------
        area : tuple, optional
            The north-west-south-east boundaries of the mask
        """
        
        self.area = area

        self._mask = None
        
        # coordinates of the original (non-cropped) fields
        self.origin_latitudes = None
        self.origin_longitudes = None
        
        # coordinates of the cropped fields
        self._latitudes = None
        self._longitudes = None

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Crop each of the fields with a mask deduced from the first field.

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
            first = fields[0]
            data = first.to_numpy(flatten=True)

            self.origin_latitudes, self.origin_longitudes = first.grid_points()
            
            self._mask = cropping_mask(
                self.origin_latitudes,
                self.origin_longitudes,
                area['north'],
                area['west'],
                area['south'],
                area['east']
            )
            
            self._latitudes = self.origin_latitudes[self._mask]
            self._longitudes = self.origin_longitudes[self._mask]

        result = []
        for field in tqdm.tqdm(fields, desc="Cropping with Mask"):

            data = field.to_numpy(flatten=True)
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(data[self._mask], template=field),
                    latitudes=self._latitudes,
                    longitudes=self._longitudes,
                )
            )

        return new_fieldlist_from_list(result)
