# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np
import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter


class RadianceToBrightnessTemperature(TabularFilter, registry_name="radiance_to_brightness_temperature"):
    """Convert CrIS NSR/FSR radiances (mW/(m^2·sr·cm^-1)) to brightness temperatures [K].

    The config should contain the following keys:
        - input_prefix: prefix of the input column names
        - output_prefix: prefix of the output column names
        - mode: 'cris_fsr' or 'cris_nsr'

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - radiance_to_brightness_temperature:
              input_prefix: obsvalue_rad_
              output_prefix: obsvalue_rawbt_
              mode: cris_fsr

    """

    def __init__(self, *, mode: str, input_prefix: str = "obsvalue_rad_", output_prefix: str = "obsvalue_rawbt_"):
        if mode not in ("cris_fsr", "cris_nsr"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'cris_fsr' or 'cris_nsr'.")
        self.mode = mode
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix

    @staticmethod
    def _cris_fsr_wavenumbers(ch):
        """Vectorised CrIS FSR channel -> wavenumber [cm^-1]"""
        ch = np.asarray(ch, dtype=np.int64)
        nu = np.full(ch.shape, np.nan, dtype=float)

        m1 = (1 <= ch) & (ch <= 713)
        m2 = (714 <= ch) & (ch <= 1578)
        m3 = (1579 <= ch) & (ch <= 2211)

        nu[m1] = 650.0 + 0.625 * (ch[m1] - 1)
        nu[m2] = 1210.0 + 0.625 * (ch[m2] - 714)
        nu[m3] = 2155.0 + 0.625 * (ch[m3] - 1579)
        return nu

    @staticmethod
    def _cris_nsr_wavenumbers(ch):
        """Vectorised CrIS NSR channel -> wavenumber [cm^-1]"""
        ch = np.asarray(ch, dtype=np.int64)
        nu = np.full(ch.shape, np.nan, dtype=float)

        m1 = (1 <= ch) & (ch <= 713)
        m2 = (714 <= ch) & (ch <= 1146)
        m3 = (1147 <= ch) & (ch <= 1305)

        nu[m1] = 650.0 + 0.625 * (ch[m1] - 1)
        nu[m2] = 1210.0 + 1.250 * (ch[m2] - 714)
        nu[m3] = 2155.0 + 2.500 * (ch[m3] - 1147)
        return nu

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        # Pick columns and parse channel numbers (vectorised)
        cols_mask = obs_df.columns.str.startswith(self.input_prefix)
        cols = obs_df.columns[cols_mask]
        if len(cols) == 0:
            raise ValueError(f"No columns starting with '{self.input_prefix}' found in DataFrame.")

        chans = obs_df.columns[cols_mask].str.extract(r"_(\d+)$", expand=False).astype(int).to_numpy()

        # Sort by channel to align wavenumbers with columns
        order = np.argsort(chans)
        cols = cols[order]
        chans = chans[order]

        C1_W = 1.191042e-10  # W m^-2 sr^-1 cm^3
        C2 = 1.4387768775  # K cm

        # Wavenumbers and Planck coefficients
        if self.mode == "cris_fsr":
            nu = self._cris_fsr_wavenumbers(chans)  # [cm^-1]
        elif self.mode == "cris_nsr":
            nu = self._cris_nsr_wavenumbers(chans)  # [cm^-1]
        else:
            logging.error(f"radiance_to_brightness_temperature ERROR: Mode not supported {self.mode}")

        a = C1_W * (nu**3)  # shape (nch,)
        b = C2 * nu

        # Radiance matrix in W/(m^2·sr·cm^-1), then invert Planck
        R = obs_df[cols].to_numpy(dtype=float, copy=False) * 1e-2

        Tb = b / np.log1p(a / np.maximum(R, 1e-300))  # (nrow, nch)

        # Write back and rename
        out_df = obs_df.copy()
        out_df.loc[:, cols] = Tb
        out_df = out_df.rename(columns={c: f"{self.output_prefix}{ch}" for c, ch in zip(cols, chans)})

        return out_df
