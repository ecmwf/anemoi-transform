# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


@filter_registry.register("encode_statids")
class EncodeStatids(TabularFilter):
    """Encode the station ID column of a DataFrame as an integer using hash as a
    fallback.

    The configuration of this filter can contain a `station_id` key to specify
    the column name to encode. If not provided, defaults to 'statid'.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - encode_statids:
              station_id: statid

    """

    def __init__(self, *, station_id: str = "statid"):
        self.station_id = station_id

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, [self.station_id])

        def statid_to_int(station_id: str) -> int:
            import hashlib

            """Convert station ID to integer with hash fallback"""
            s = station_id.strip().upper()

            # Check if valid base-36
            if s and all(c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in s):
                return int(s, 36)
            else:
                # Use hash for invalid characters
                h = hashlib.md5(station_id.strip().encode()).digest()
                return int.from_bytes(h[:4], "little", signed=False)

        obs_df[self.station_id] = obs_df[self.station_id].apply(statid_to_int)
        return obs_df
