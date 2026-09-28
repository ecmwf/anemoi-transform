# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""GRIB-specific routines built on the earthkit-data facade.

Everything that touches the GRIB machinery of earthkit-data (encoders,
codes handles) lives here, so that :mod:`anemoi.transform.fields` stays
format-agnostic and GRIB knowledge is confined to a single module. Together
with :mod:`anemoi.transform.fields` this is the only place in the anemoi
packages allowed to import ``earthkit.data``.
"""

import logging
from typing import Any

from anemoi.transform.fields import _unwrap_field

LOG = logging.getLogger(__name__)


class _GribOutput:
    """A minimal GRIB file writer.

    earthkit-data 1.0 removed ``earthkit.data.readers.grib.output.new_grib_output``.
    This reproduces the small subset the Anemoi packages depend on
    (``write`` / ``close``) on top of the 1.0 :class:`~earthkit.data.encoders.grib.GribEncoder`.
    """

    def __init__(self, path: str):
        """Open ``path`` for writing GRIB messages."""
        self._file = open(path, "wb")

    def write(
        self,
        values: Any,
        check_nans: bool = True,
        metadata: dict | None = None,
        template: Any = None,
        missing_value: float = 9999,
        **kwargs: Any,
    ) -> None:
        """Encode one GRIB message and append it to the file.

        Parameters
        ----------
        values : Any
            The values to encode.
        check_nans : bool
            Replace NaNs in the values with ``missing_value``.
        metadata : dict, optional
            Metadata to encode; merged with ``**kwargs``.
        template : Field, optional
            A (wrapped or raw) field used as encoding template.
        missing_value : float
            The value encoded in place of NaNs.
        **kwargs : Any
            Additional metadata to encode.
        """
        from earthkit.data.encoders.grib import GribEncoder

        metadata = {**(metadata or {}), **kwargs}
        GribEncoder().encode(
            values=values,
            template=_unwrap_field(template) if template is not None else None,
            check_nans=check_nans,
            metadata=metadata,
            missing_value=missing_value,
        ).to_file(self._file)

    def close(self) -> None:
        """Close the output file."""
        self._file.close()


def new_grib_output(path: str) -> _GribOutput:
    """Open a GRIB file for writing (earthkit-data 1.0 compatible).

    Drop-in replacement for the removed
    ``earthkit.data.readers.grib.output.new_grib_output``.

    Parameters
    ----------
    path : str
        The path of the GRIB file to create.

    Returns
    -------
    _GribOutput
        The opened GRIB writer (``write`` / ``close``).
    """
    return _GribOutput(path)


def grib_handle(field: Any) -> Any:
    """Return the GRIB codes handle backing a field.

    Encapsulates the earthkit-data private API used to reach the raw
    handle of a GRIB-backed field, so that consumers (e.g. the
    anemoi-inference GRIB encoder, which sets interdependent keys directly
    on a cloned handle) do not touch earthkit-data internals themselves.

    Parameters
    ----------
    field : Field
        The (wrapped or raw) field whose handle is requested. Must be
        backed by a GRIB message.

    Returns
    -------
    earthkit.data.readers.grib.handle.GribCodesHandle
        The handle of the field's GRIB message. Clone it before mutating.

    Raises
    ------
    ValueError
        If the field is not backed by a GRIB message.
    """
    grib = _unwrap_field(field)._get_grib(strict=True)
    if grib is None:
        raise ValueError(f"Field is not backed by a GRIB message: {field}")
    return grib.handle
