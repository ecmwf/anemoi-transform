# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os.path
from typing import List

import pandas as pd
from earthkit.data.readers.odb import ODBReader
from . import filter_registry
from .base import SimpleFilter

INDEX_COL = "seqno@hdr"
GEOLOCATION_META_COLS = ["lat@hdr", "lon@hdr", "date@hdr", "time@hdr"]
VARNO_COL = "varno@body"


class ReshapeODBDF(SimpleFilter):
    """A filter to reshape ODB dataframe."""

    def __init__(
        self,
        *,
        sort_by: List[str] = ["date@hdr", "time@hdr"],
        meta_cols: List[str] = [],
        meta_body_cols: List[str] = [],
        extra_obsval_cols: List[str] = [],
        predicted_cols: List[str] = ["obsvalue@body"],
        pivot_cols: List[str] = ["varno@body"],
        drop_nans: bool = False,
    ):
        self.sort_by = sort_by
        self.meta_cols = meta_cols
        self.meta_body_cols = meta_body_cols
        self.extra_obsval_cols = extra_obsval_cols
        self.predicted_cols = predicted_cols
        self.pivot_cols = pivot_cols
        self.drop_nans = drop_nans

        if not all([self.predicted_cols, self.pivot_cols]):
            raise ValueError("'predicted_col' and 'pivot_col' must be specified")

    def forward(self, data):
        yield self._transform(data, self.forward_transform)

    def backward(self, data):
        raise NotImplementedError("ReshapeODBDF is not reversible")

    def forward_transform(self, data: ODBReader) -> pd.DataFrame:
        """
        Restructures a dataframe in the native ODB-schema
            i) pivot so that per-channel or per-variable from row-wise to column-wise
            ii) renames columns
            iii) sorts the data
        """
        index_cols = [INDEX_COL] + GEOLOCATION_META_COLS + self.meta_cols
        value_cols = self.predicted_cols + self.meta_body_cols

        pivot_colname = (
            ["varno@body", "vertco_reference_1@body"]
            if self.pivot_cols == ["vertco_reference_1@body"]
            else self.pivot_cols
        )

        df = data.to_pandas()
        df = df.drop_duplicates(subset=index_cols + pivot_colname, keep="first")

        df_pivot = df.pivot(index=index_cols, columns=pivot_colname, values=value_cols)
        df_pivot = df_pivot.sort_values(by=self.sort_by, kind="stable").reset_index()

        df_meta = df_pivot[index_cols]
        df_obs = df_pivot.drop(columns=index_cols, level=0).sort_index(axis=1)
        df_out = pd.concat([df_meta, df_obs], axis=1)

        if self.drop_nans:
            df_out = df_out.dropna()

        df_out["datetime"] = pd.to_datetime(
            df_out["date@hdr"].astype(int).astype(str)
            + df_out["time@hdr"].astype(int).astype(str).str.zfill(6),
            format="%Y%m%d%H%M%S",
        )
        df_out = df_out.drop(columns=["date@hdr", "time@hdr"], level=0)

        df_out.columns = self.rename_columns(df_out.columns.tolist(), self.extra_obsval_cols)

        return df_out

    def rename_columns(self, tup_list: List, extra_obsval_cols: List[str]) -> List[str]:
        """
        Rename the columns using convention: obsvalue_{varno_name}_{vertco_reference_1}
        Note: non-obsvalue columns simply have the "@table" stripped from the name.

        Args:
            tup_list: List of tuples from pandas multi-index column names
                e.g. ("obsvalue@body",39) -> "obsvalue_t2m_0"
                     ("obsvalue@body",119,22) -> "obsvalue_rawbt_22"
            extra_obsval_cols: List of additional column names to be treated as observation values

        Returns:
            List of new column names
        """
        path = os.path.dirname(os.path.abspath(__file__))
        with open(f"{path}/../data/varno.json") as f:
            varno_dict = json.load(f)

        out_colnames = []
        for tup in tup_list:
            colname = tup[0]
            varno = tup[1] if len(tup) > 1 else ""
            vertco_reference_1 = tup[2] if len(tup) > 2 else ""

            base_colname = colname.split("@")[0]

            if base_colname in extra_obsval_cols:
                base_colname = f"obsvalue_{base_colname}"

            if not varno:
                out_colnames.append(base_colname)
            else:
                try:
                    varno_idx = next(
                        i
                        for i, varno_lst in enumerate(varno_dict["data"])
                        if int(varno) in varno_lst
                    )
                    varno_name = varno_dict["data"][varno_idx][0]
                    vertco_suffix = (
                        f"{int(vertco_reference_1)}" if vertco_reference_1 else "0"
                    )
                    out_colnames.append(f"{base_colname}_{varno_name}_{vertco_suffix}")
                except (ValueError, StopIteration):
                    logging.warning(
                        f"Unable to find varno name for {varno}. Using original varno."
                    )
                    out_colnames.append(f"{base_colname}_{varno}_{vertco_suffix}")

        return out_colnames

    def backward_transform(self, data: ODBReader) -> None:
        raise NotImplementedError("ReshapeODBDF is not reversible")


filter_registry.register("reshape_odb_df", ReshapeODBDF)
