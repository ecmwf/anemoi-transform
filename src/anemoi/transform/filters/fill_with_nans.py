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

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("fill_with_nans")
class FillWithNaNs(Filter):

    def __init__(self):
        self._mask = None

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:

        return fields

    def find_idxs(self, vec, new_vec):
        import numpy as np
        idx = 0
        idx_table = []
        print(f"shape of vec {vec.shape} and new_vec {new_vec.shape}")
        for new_idx in range(len(new_vec)):
            
            err = 0
            new_value = new_vec[new_idx]
            
            try:
                value = vec[idx]
            except IndexError:
                idx-=1
                value = vec[idx]
                err = 1

            if np.isclose(new_value, value):
                # print("\nnew_idx", new_idx)
                # if err:
                #     print("idx ceiled")
                # print("idx", idx)
                idx_table.append(idx)
                idx+=1
            else:
                idx_table.append(-1)

        print(f"final idx = {idx}")
        print(f"final nex_idx = {new_idx}")
        return idx_table

    def backward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """

         Parameters
        ----------
        fields : ekd.FieldList
            List of fields to be processed.

        Returns
        -------
        ekd.FieldList

        """
        import numpy as np
        from itertools import groupby
        import tqdm
        fill_value = 9999

        if self._mask is None:

            first = fields[0]
            lon, lat = first.state['longitudes'], first.state['latitudes']
            data = first.to_numpy(flatten=True)

            max_lon, min_lon = max(lon), min(lon)
            max_lat, min_lat = max(lat), min(lat)
            
            list_nb_lon_by_lat = [len(list(g)) for v, g in groupby(lat)]
            print("nb_lon_by_lat", list_nb_lon_by_lat)
            cumsum = np.cumsum(list_nb_lon_by_lat)
            sum_nb_lon_before_lat = np.insert(cumsum[:-1], 0, 0)

            unique_lons = np.unique(lon)
            unique_lats = np.unique(lat)
            nb_lats = len(unique_lats)
            nb_lons = len(unique_lons)

            step_lon = unique_lons[1] - unique_lons[0]
            step_lat = unique_lats[1] - unique_lats[0]

            new_lon = np.tile(np.arange(min_lon, max_lon + step_lon, step_lon), nb_lats)
            new_lat = np.repeat(np.arange(min_lat, max_lat + step_lat, step_lat), nb_lons)[::-1]


            idx_lons = ((lon - min_lon)/step_lon).astype(int)

            new_data = np.ones((nb_lats * nb_lons)) * fill_value
            print("shape", new_data.shape)
            result = []
            for field in tqdm.tqdm(fields, desc=f"Fill with {fill_value}"):
                _lon, _lat = field.state['longitudes'], field.state['latitudes']
                for idx_lat in range(nb_lats):
                    nb_lon_before_lat = sum_nb_lon_before_lat[idx_lat]
                    nb_lon_by_lat = list_nb_lon_by_lat[idx_lat]
                    list_idx = idx_lat*nb_lats+ idx_lons[nb_lon_before_lat : nb_lon_before_lat + nb_lon_by_lat]
                    new_data[list_idx] = data[nb_lon_before_lat: nb_lon_before_lat + nb_lon_by_lat]
                result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(new_data, template=field),
                    latitudes=new_lat,
                    longitudes=new_lon,
                )
            )
                
            print("len results : ", len(result))
            print("end ",  new_fieldlist_from_list(result))

            return new_fieldlist_from_list(result)
            # print("new_data : ", new_data.shape)
            # print("new_data : ", new_data)


            # print("\n", dir(first))
            # print(type(fields))
            # print(type(first))
            # print(first.metadata())
            # print(first.state)


            # print("lon:\n", lon)
            # print("lon:\n", lon.shape)
            # print("lat:\n", lat)
            # print("lat:\n", lat.shape)

            # print("new_lon:\n", new_lon.shape)
            # print("new_lat:\n", new_lat.shape)
            # print("new_lon:\n", new_lon)
            # print("new_lat:\n", new_lat)

            # print("Finding lon idx")
            # idx_lon_table = self.find_idxs(lon, new_lon)
            # print("Finding lat idx")
            # print("lat : ", lat[0:10])
            # print("lat : ", new_lat[0:10])
            # idx_lat_table = self.find_idxs(lat, new_lat)
            
            # print("idx lon shape : ", len(idx_lon_table))
            # print("idx lat shape : ", len(idx_lat_table))

            # print("sum", len(np.where((idx_lon_table == -1) & (idx_lat_table == -1))[0]))

            # print("idx lon : ", idx_lon_table)
            # print("idx lat : ", idx_lat_table)



        # for field in fields:
        #     data = field.to_numpy(flatten=True)

        #     print("\nfield : ", field)
        #     print(data.shape)



        # return fields
    
