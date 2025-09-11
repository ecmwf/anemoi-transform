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

import numpy as np
from itertools import groupby
import tqdm

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("fill_with_nans")
class FillWithNaNs(Filter):

    def __init__(
            self, 
            fill_value=9999, 
            max_lon_output = 16.0, 
            min_lon_output = -12.0,  
            max_lat_output = 55.4, 
            min_lat_output = 37.5
            ):
        self.fill_value = fill_value
        self.max_lon_output = max_lon_output
        self.min_lon_output = min_lon_output
        self.max_lat_output = max_lat_output
        self.min_lat_output = min_lat_output

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:

        return fields

    def backward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """
        Fill missing grid points with a default value in the fields.
        The step is supposed to be the same
         Parameters
        ----------
        fields : ekd.FieldList
            List of fields to be processed.

        Returns
        -------
        ekd.FieldList

        """

        first = fields[0]
        input_lon, input_lat = first.state['longitudes'], first.state['latitudes']
        print("input_lat", input_lat)
        unique_lons = np.unique(input_lon)
        unique_lats = np.unique(input_lat)
        nb_lats_input = len(unique_lats)

        step_lon = unique_lons[1] - unique_lons[0]
        step_lat = unique_lats[1] - unique_lats[0]

        nb_lats_output = round((self.max_lat_output - self.min_lat_output) / step_lat)
        nb_lons_output = round((self.max_lon_output - self.min_lon_output) / step_lon)
        print("nb_lats_desired", nb_lats_output)
        print("nb_lons_desired", nb_lons_output)
        output_lon = np.tile(np.arange(self.min_lon_output, self.max_lon_output + step_lon, step_lon), nb_lats_output)
        output_lat = np.repeat(np.arange(self.min_lat_output, self.max_lat_output + step_lat, nb_lats_output), nb_lons_output)[::-1]
        print("new_lon", output_lon)
        print("new_lat", output_lat)

        list_nb_lon_by_lat_in_input = [len(list(g)) for v, g in groupby(input_lat)]
        print("nb_lon_by_lat", list_nb_lon_by_lat_in_input)
        cumsum = np.cumsum(list_nb_lon_by_lat_in_input)
        sum_nb_lon_before_lat_in_input = np.insert(cumsum[:-1], 0, 0)

        list_idx_output_lon = np.rint((input_lon - self.min_lon_output)/step_lon).astype(int)
        list_idx_output_lat = np.rint((self.max_lat_output - input_lat)/step_lat).astype(int)
        print("aaa",np.unique(list_idx_output_lat) )
        output_data = np.ones((nb_lats_output * nb_lons_output)) * self.fill_value
        result = []
        # for field in tqdm.tqdm(fields, desc=f"Fill with {self.fill_value}"):
            # _lon, _lat = len(np.unique(field.state['longitudes'])), len(np.unique(field.state['latitudes']))
            # print(f"size fields lon lat {_lon} and {_lat}")
        field = fields[0]
        input_data = field.to_numpy(flatten=True)
        for idx_input_lat in range(nb_lats_input):
            print("\nlatitude idx dans l'ancien tableau", idx_input_lat)
            nb_lon_before_lat_in_input = sum_nb_lon_before_lat_in_input[idx_input_lat]
            print(f"Il y avait {nb_lon_before_lat_in_input} points avant cette ligne")
            nb_lon_by_lat_in_input = list_nb_lon_by_lat_in_input[idx_input_lat]
            print(f"Il y a {nb_lon_by_lat_in_input} lon à cette latitude")
            idx_output_lat = list_idx_output_lat[nb_lon_before_lat_in_input : nb_lon_before_lat_in_input + nb_lon_by_lat_in_input][0]
            print("idx latitude dnas le nouveau tableau : ", idx_output_lat)
            list_idx = idx_output_lat*nb_lats_output + list_idx_output_lon[nb_lon_before_lat_in_input : nb_lon_before_lat_in_input + nb_lon_by_lat_in_input]
            print("idx lat dans output_data qu'on va écrire", idx_output_lat)
            print("idx lon dans output_data qu'on va écrire", list_idx_output_lon[nb_lon_before_lat_in_input : nb_lon_before_lat_in_input + nb_lon_by_lat_in_input])
            print("idx dans input data qu'on préleve", nb_lon_before_lat_in_input,  nb_lon_before_lat_in_input + nb_lon_by_lat_in_input)
            output_data[list_idx] = input_data[nb_lon_before_lat_in_input: nb_lon_before_lat_in_input + nb_lon_by_lat_in_input]
            print("output_data[list_idx]", output_data[list_idx])

        result.append(
        new_field_from_latitudes_longitudes(
            new_field_from_numpy(output_data, template=field),
            latitudes=output_lat,
            longitudes=output_lon,
        )
    )
            
        print("len results : ", len(result))
        print("end ",  new_fieldlist_from_list(result))

        return new_fieldlist_from_list(result)
        
