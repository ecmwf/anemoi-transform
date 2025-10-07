# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from itertools import groupby

import earthkit.data as ekd
import numpy as np
import tqdm

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("fill_square_gribs")
class FillSquareGribs(Filter):
    """A filter to recenter gribs at given coordinates
    Fill the missing values with a default value.
    """

    def __init__(
        self, fill_value=9999, max_lon_output=16.0, min_lon_output=-12.0, max_lat_output=55.4, min_lat_output=37.5
    ):
        """Initialize the FillSquareGribs filter.

        Parameters
        ----------
        fill_value : int
            The default value to use to fill gribs
        max_lon_output : float
            The maximal longitude of the square
        min_lon_output : float
            The minimal longitude of the square
        max_lat_output : float
            The maximal latitude of the square
        min_lat_output : float
            The minimal latitude of the square
        """

        self.fill_value = fill_value
        self.max_lon_output = max_lon_output
        self.min_lon_output = min_lon_output
        self.max_lat_output = max_lat_output
        self.min_lat_output = min_lat_output

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:

        return fields

    def backward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Fill missing grid points with a default value in the fields.
        The longitude step and the latitude step is supposed to be constant.
         Parameters
        ----------
        fields : ekd.FieldList
            List of fields to be processed.

        Returns
        -------
        ekd.FieldList

        """
        first = fields[0]
        input_lon, input_lat = first.state["longitudes"], first.state["latitudes"]
        input_data = first.to_numpy(flatten=True)
        print(input_data.shape)
        unique_lons = np.unique(input_lon)
        unique_lats = np.unique(input_lat)
        print(unique_lons)
        print(unique_lats)
        print(len(unique_lons))
        print(len(unique_lats))

        nb_lats_input = len(unique_lats)

        step_lon = unique_lons[1] - unique_lons[0]
        step_lat = unique_lats[1] - unique_lats[0]
        nb_lats_output = round((self.max_lat_output - self.min_lat_output) / step_lat) + 1
        nb_lons_output = round((self.max_lon_output - self.min_lon_output) / step_lon) + 1

        output_lon = np.tile(np.arange(self.min_lon_output, self.max_lon_output + step_lon, step_lon), nb_lats_output)
        output_lat = np.repeat(
            np.arange(self.min_lat_output, self.max_lat_output + step_lat, nb_lats_output), nb_lons_output
        )[::-1]

        # Number of lon at the latitude idx
        # list_nb_lon_by_lat_in_input = [201, 291 ...]
        list_nb_lon_by_lat_in_input = [len(list(g)) for v, g in groupby(input_lat)]

        # Each element of the vector is the sum of the number of longitude before a given latitude in the data
        # First element represents the first latitude, there is 0 longitude before the first lat
        # sum_nb_lon_before_lat_in_input = [0, 201, 492 ...]
        cumsum = np.cumsum(list_nb_lon_by_lat_in_input)
        sum_nb_lon_before_lat_in_input = np.insert(cumsum[:-1], 0, 0)

        # Compute longitude and latitude idx in the input vector
        # list_idx_output_lon = [0, 1, 2, ..., 1120, 0, 1, ..., 1120 ...]
        list_idx_output_lon = np.rint((input_lon - self.min_lon_output) / step_lon).astype(int)
        # latitude are reversed so the index is computed from the max
        list_idx_output_lat = np.rint((self.max_lat_output - input_lat) / step_lat).astype(int)

        result = []
        for field in tqdm.tqdm(fields, desc=f"Fill with {self.fill_value}"):
            input_data = field.to_numpy(flatten=True)
            output_data = np.ones((nb_lats_output, nb_lons_output)) * self.fill_value
            output_data[list_idx_output_lat, list_idx_output_lon] = input_data
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(output_data, template=field),
                    latitudes=output_lat,
                    longitudes=output_lon,
                )
            )

        return new_fieldlist_from_list(result)
