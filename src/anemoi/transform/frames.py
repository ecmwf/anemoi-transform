# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import os
from typing import Any

import pandas as pd
from pandas import DataFrame as _PandasFrame

from anemoi.transform.datum import Datum

LOG = logging.getLogger(__name__)


def _unwrap_frame(frame: "Frame | _PandasFrame") -> _PandasFrame:
    """Return the underlying pandas DataFrame for either a wrapped or raw frame."""
    return frame._frame if isinstance(frame, Frame) else frame


class Frame(Datum):
    """A thin, transparent wrapper around a pandas DataFrame.

    This mirrors the way :class:`anemoi.transform.fields.FieldList` wraps an
    earthkit-data field list. Attribute access that is not explicitly defined
    here is delegated to the underlying pandas DataFrame, so accessors such as
    ``columns``, ``shape``, ``dtypes``, ``loc`` and ``iloc`` and methods such as
    ``head`` remain available.
    """

    def __init__(self, frame: _PandasFrame | None = None):
        self._frame = frame if frame is not None else pd.DataFrame()

    @property
    def _underlying(self) -> _PandasFrame:
        return self._frame

    def to_pandas(self) -> _PandasFrame:
        """Return the underlying pandas DataFrame."""
        return self._frame

    @classmethod
    def from_pandas(cls, frame: "Frame | _PandasFrame") -> "Frame":
        """Create a Frame from an existing pandas DataFrame (or another Frame)."""
        return cls(_unwrap_frame(frame))

    @classmethod
    def from_dicts(cls, dicts: list[dict]) -> "Frame":
        """Create a Frame from a list of dictionaries."""
        return cls(pd.DataFrame.from_records(dicts))

    @classmethod
    def from_file(cls, path: str, **kwargs: Any) -> "Frame":
        """Create a Frame from a file, inferring the format from its extension.

        Supported extensions: ``.csv``, ``.parquet``, ``.feather``, ``.json``
        and ``.pkl``/``.pickle``.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".csv":
            return cls.from_csv(path, **kwargs)
        if ext == ".parquet":
            return cls.from_parquet(path, **kwargs)
        if ext == ".feather":
            return cls(pd.read_feather(path, **kwargs))
        if ext == ".json":
            return cls(pd.read_json(path, **kwargs))
        if ext in (".pkl", ".pickle"):
            return cls(pd.read_pickle(path, **kwargs))
        raise ValueError(f"Cannot infer frame format from file extension '{ext}' for path '{path}'.")

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "Frame":
        """Create a Frame from a CSV file."""
        return cls(pd.read_csv(path, **kwargs))

    @classmethod
    def from_parquet(cls, path: str, **kwargs: Any) -> "Frame":
        """Create a Frame from a Parquet file."""
        return cls(pd.read_parquet(path, **kwargs))

    @classmethod
    def concat(cls, *args: "Frame | _PandasFrame", **kwargs: Any) -> "Frame":
        """Concatenate multiple Frames into a single Frame."""
        return cls(pd.concat([_unwrap_frame(arg) for arg in args], **kwargs))

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, key: Any) -> Any:
        return self._frame[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._frame[key] = value

    def __iter__(self):
        return iter(self._frame)

    def __contains__(self, key: Any) -> bool:
        return key in self._frame

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._frame!r})"
