# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import healpy as hp
import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter


class AddHealpix(TabularFilter, registry_name="add_healpix"):
    """Add a healpix index column 'healpix_idx_{nside}' to the DataFrame.

    The configuration key `nside` is an integer representing the number of
    pixels per side (must be positive).

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - add_healpix:
              nside: 32

    """

    def __init__(self, *, nside: int = 32):
        if nside <= 0:
            raise ValueError("nside must be a positive integer.")
        self.nside = nside

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        obs_df[f"healpix_idx_{self.nside}"] = hp.ang2pix(
            self.nside,
            obs_df["longitude"].values,
            obs_df["latitude"].values,
            nest=True,
            lonlat=True,
        )
        return obs_df
