# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import earthkit.data as ekd
import numpy as np
from earthkit.data import Field as _EkdField
from earthkit.data import FieldList as _EkdFieldList
from earthkit.data.core.order import build_remapping
from earthkit.data.utils.dates import to_datetime
from earthkit.data.utils.dates import to_timedelta

from anemoi.transform.datum import Datum

LOG = logging.getLogger(__name__)

# Sentinel returned by a Flavour when it has no value for a given metadata key.
MISSING_METADATA = object()

# ---------------------------------------------------------------------------
# earthkit-data facade
#
# This module is the single place in the Anemoi packages that imports
# ``earthkit.data``. Everything the rest of the codebase needs from
# earthkit-data is re-exported here (see ``fields``, ``concat``,
# ``from_source``, ``Pattern``, ``build_remapping``, ``to_datetime``, ... and
# the lazily-loaded names in ``__getattr__`` below). Import from
# ``anemoi.transform.fields`` rather than adding a new ``earthkit.data`` import
# elsewhere.
# ---------------------------------------------------------------------------

# Raw earthkit-data types, exposed under explicit names so they do not clash
# with the wrapper ``Field`` / ``FieldList`` classes defined below (which are
# the intended abstraction for gridded data collections).
EarthkitField = _EkdField
EarthkitFieldList = _EkdFieldList

# earthkit 1.0rc12 compatibility shim (moved here from anemoi-datasets so that
# earthkit-data internals are only ever touched in this module):
# ``earthkit.data.utils.unique.build_remapping`` always returns ``None`` (bug).
# Patch it to actually return the built remapping. Remove when fixed upstream.
import earthkit.data.utils.unique as _ekd_unique  # noqa: E402

if not getattr(_ekd_unique, "_build_remapping_patched", False):

    def _patched_unique_build_remapping(remapping: Any, patch: Any) -> Any:
        if remapping is not None or patch is not None:
            return build_remapping(remapping, patch)
        return None

    _ekd_unique.build_remapping = _patched_unique_build_remapping
    _ekd_unique._build_remapping_patched = True


class _GribOutput:
    """A minimal GRIB file writer.

    earthkit-data 1.0 removed ``earthkit.data.readers.grib.output.new_grib_output``.
    This reproduces the small subset the Anemoi packages depend on
    (``write`` / ``close``) on top of the 1.0 :class:`~earthkit.data.encoders.grib.GribEncoder`.
    """

    def __init__(self, path: str):
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
        from earthkit.data.encoders.grib import GribEncoder

        metadata = {**(metadata or {}), **kwargs}
        GribEncoder().encode(
            values=values,
            template=template,
            check_nans=check_nans,
            metadata=metadata,
            missing_value=missing_value,
        ).to_file(self._file)

    def close(self) -> None:
        self._file.close()


def new_grib_output(path: str) -> _GribOutput:
    """Open a GRIB file for writing (earthkit-data 1.0 compatible).

    Drop-in replacement for the removed
    ``earthkit.data.readers.grib.output.new_grib_output``.
    """
    return _GribOutput(path)


def __getattr__(name: str) -> Any:
    """Lazily re-export heavier / optional earthkit-data internals.

    Kept out of the eager import block so that importing
    :mod:`anemoi.transform.fields` stays cheap; the underlying module is only
    imported on first access (PEP 562).
    """
    _lazy = {
        "Availability": ("earthkit.data.utils.availability", "Availability"),
        "temp_file": ("earthkit.data.core.temporary", "temp_file"),
        "GribEncoder": ("earthkit.data.encoders.grib", "GribEncoder"),
        "XArrayFieldList": ("earthkit.data.readers.xarray.fieldlist", "XArrayFieldList"),
        "download_and_cache": ("earthkit.data.sources.url", "download_and_cache"),
    }
    if name in _lazy:
        import importlib

        module_name, attr = _lazy[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _unwrap_field(field: "Field | _EkdField") -> _EkdField:
    """Return the underlying earthkit field for either a wrapped or raw field."""
    return field._field if isinstance(field, Field) else field


def _unwrap_fieldlist(fieldlist: "FieldList | _EkdFieldList") -> _EkdFieldList:
    """Return the underlying earthkit fieldlist for either a wrapped or raw fieldlist."""
    return fieldlist._fieldlist if isinstance(fieldlist, FieldList) else fieldlist


class Field:
    """A thin, transparent wrapper around an earthkit-data field.

    Attribute access that is not explicitly defined here is delegated to the
    underlying earthkit field, so component accessors such as ``parameter``,
    ``time``, ``geography``, ``vertical`` and ``ensemble`` remain available.
    """

    def __init__(self, field: _EkdField | None = None):
        self._field = field

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails.
        # Delegate to the underlying earthkit field.
        if name == "_field":
            raise AttributeError(name)
        return getattr(self._field, name)

    # === forwarded methods for Field class

    def set(self, *args, **kwargs) -> "Field":
        return Field(self._field.set(*args, **kwargs))

    def get(self, *args, **kwargs) -> Any:
        return self._field.get(*args, **kwargs)

    def to_numpy(self, *args, **kwargs) -> Any:
        return self._field.to_numpy(*args, **kwargs)

    # ===
    @classmethod
    def from_components(cls, **kwargs: Any) -> "Field":
        """Create a new field from its components.

        This is a thin wrapper around :meth:`earthkit.data.Field.from_components`
        that returns a wrapped :class:`Field`.

        Parameters
        ----------
        **kwargs : Any
            The components of the field (e.g. ``values``, ``parameter``,
            ``geography``, ``time``, ``labels``, ...).

        Returns
        -------
        Field
            The new field created from the given components.
        """
        return cls(_EkdField.from_components(**kwargs))

    @staticmethod
    def new_grib_handle(handle: Any) -> Any:
        """Create a new earthkit GRIB codes handle.

        Parameters
        ----------
        handle : Any
            A raw eccodes handle to wrap.

        Returns
        -------
        earthkit.data.readers.grib.handle.GribCodesHandle
            The new GRIB codes handle.
        """
        from earthkit.data.readers.grib.handle import GribCodesHandle

        return GribCodesHandle(handle, None, None)

    @classmethod
    def from_numpy(cls, array: np.ndarray, *, template: "Field", **metadata: Any) -> "Field":
        """Create a new field from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            The data for the new field.
        template : Field
            The template field to use.
        **metadata : Any
            Additional metadata for the new field.

        Returns
        -------
        Field
            The new field created from the numpy array and template.
        """
        result = cls(_unwrap_field(template).set(**{"data.values": array}))
        if metadata:
            result = cls.with_new_metadata(result, **metadata)

        return result

    @classmethod
    def with_new_metadata(cls, template: "Field", **metadata: Any) -> "Field":
        """Create a new field with metadata.

        Parameters
        ----------
        template : Field
            The template field to use.
        **metadata : Any
            The metadata for the new field.

        Returns
        -------
        Field
            The new field with the provided metadata.
        """
        key_mapping = {
            "valid_datetime": "time.valid_datetime",
            "base_datetime": "time.base_datetime",
            "step": "time.step",
            "param": "parameter.variable",
            "units": "parameter.units",
            "levtype": "vertical.level_type",
            "levelist": "vertical.level",
            "number": "ensemble.member",
        }

        unknown_keys = set(metadata.keys()) - set(key_mapping.keys())
        if unknown_keys:
            raise ValueError(f"Unknown metadata keys: {unknown_keys}. Allowed keys are: {set(key_mapping.keys())}")

        # map metadata keys to new locations
        mapped_metadata = {key_mapping[key]: value for key, value in metadata.items()}
        return cls(_unwrap_field(template).set(**mapped_metadata))

    @classmethod
    def with_valid_datetime(cls, template: "Field", date: Any) -> "Field":
        """Create a new field with a valid datetime (sets the step to 0).

        Setting the ``step`` to 0 means the ``base_datetime`` is updated to be
        equal to the new ``valid_datetime``.

        Parameters
        ----------
        template : Field
            The template field to use.
        date : Any
            The valid datetime for the new field.

        Returns
        -------
        Field
            The new field with the provided valid datetime and a step of 0.
        """
        return cls(
            _unwrap_field(template).set(
                **{
                    "time.valid_datetime": date,
                    "time.step": datetime.timedelta(hours=0),
                }
            )
        )

    @classmethod
    def from_latitudes_longitudes(cls, template: "Field", latitudes: np.ndarray, longitudes: np.ndarray) -> "Field":
        """Create a new field from latitudes and longitudes.

        Parameters
        ----------
        template : Field
            The template field to use.
        latitudes : np.ndarray
            The latitudes for the new field.
        longitudes : np.ndarray
            The longitudes for the new field.

        Returns
        -------
        Field
            The new field with the provided latitudes and longitudes.
        """
        return cls(
            _unwrap_field(template).set(
                **{
                    "geography.latitudes": latitudes,
                    "geography.longitudes": longitudes,
                }
            )
        )

    @classmethod
    def flavoured(cls, field: "Field", flavour: "Flavour") -> "Field":
        """Create a new field whose metadata lookups are mediated by a flavour.

        Parameters
        ----------
        field : Field
            The field to wrap with the flavour.
        flavour : Flavour
            The flavour used to resolve metadata keys.

        Returns
        -------
        Field
            The new flavoured field.
        """
        raise NotImplementedError("Not implemented yet.")


class FieldList(Datum):
    """A thin, transparent wrapper around an earthkit-data fieldlist.

    Iterating or indexing a :class:`FieldList` yields :class:`Field` objects.
    Attribute access that is not explicitly defined here is delegated to the
    underlying earthkit fieldlist.
    """

    def __init__(self, fieldlist: _EkdFieldList | None = None):
        self._fieldlist = fieldlist if fieldlist is not None else ekd.create_fieldlist()
        self._fields: list[Field] | None = None

    @property
    def _underlying(self) -> _EkdFieldList:
        return self._fieldlist

    @property
    def _wrapped(self) -> list[Field]:
        if self._fields is None:
            self._fields = [f if isinstance(f, Field) else Field(f) for f in self._fieldlist]
        return self._fields

    @classmethod
    def from_fields(cls, fields: list[Field]) -> "FieldList":
        """Create a FieldList from a list of fields."""
        fields = [f if isinstance(f, Field) else Field(f) for f in fields]
        result = cls(ekd.create_fieldlist([f._field for f in fields]))
        # Preserve the identity of the provided fields.
        result._fields = fields
        return result

    @classmethod
    def from_dicts(cls, dicts: list[dict]) -> "FieldList":
        """Create a FieldList from a list of dictionaries."""
        return cls(ekd.from_source("list-of-dicts", dicts).to_fieldlist())

    @classmethod
    def from_source(cls, name: str, *args, **kwargs) -> "FieldList":
        """Create a FieldList from a source."""
        return cls(ekd.from_source(name, *args, **kwargs).to_fieldlist())

    @classmethod
    def from_file(cls, path: str) -> "FieldList":
        """Create a FieldList from a file."""
        return cls.from_source("file", path)

    @classmethod
    def concat(cls, *args: "FieldList") -> "FieldList":
        """Concatenate multiple FieldLists into a single FieldList."""
        return cls(ekd.concat(*[_unwrap_fieldlist(arg) for arg in args]).to_fieldlist())

    @staticmethod
    def to_datetime(value: Any) -> Any:
        """Convert a value to a :class:`datetime.datetime`.

        Thin wrapper around :func:`earthkit.data.utils.dates.to_datetime`.
        """

        return to_datetime(value)

    @staticmethod
    def to_timedelta(value: Any) -> Any:
        """Convert a value to a :class:`datetime.timedelta`.

        Thin wrapper around :func:`earthkit.data.utils.dates.to_timedelta`.
        """

        return to_timedelta(value)

    @staticmethod
    def availability(requests: Any) -> Any:
        """Build an :class:`earthkit.data.utils.availability.Availability`.

        Parameters
        ----------
        requests : Any
            The requests used to build the availability.

        Returns
        -------
        earthkit.data.utils.availability.Availability
            The availability object.
        """
        from earthkit.data.utils.availability import Availability

        return Availability(requests)

    def sel(self, *args, **kwargs) -> "FieldList":
        """Select a subset of the fields, returning a new :class:`FieldList`."""
        return FieldList(self._fieldlist.sel(*args, **kwargs))

    def order_by(self, *args, **kwargs) -> "FieldList":
        """Order the fields, returning a new :class:`FieldList`."""
        return FieldList(self._fieldlist.order_by(*args, **kwargs))

    def __len__(self) -> int:
        return len(self._fieldlist)

    def __getitem__(self, index: int) -> Field:
        return self._wrapped[index]

    def __iter__(self):
        return iter(self._wrapped)


class Flavour(ABC):
    @abstractmethod
    def __call__(self, key: str, field: Field) -> Any:
        """Called during field metadata lookup, so it can be modified"""
        pass


class FieldSelection:
    """A class for specifying which fields to process."""

    ALLOWED_KEYS = {"parameter.variable", "vertical.level"}

    def __init__(self, **kwargs):
        self._spec = kwargs
        self._validate_spec()
        self._sanitise_spec()
        self._all = len(self._spec) == 0

    def _validate_spec(self):
        if not set(self._spec).issubset(self.ALLOWED_KEYS):
            raise ValueError(f"Invalid keys in spec: {tuple(self._spec)} - only {self.ALLOWED_KEYS} are allowed.")

    def _sanitise_spec(self):
        for key, value in list(self._spec.items()):
            if isinstance(value, (str, int, float, bool)):
                self._spec[key] = (value,)
            elif value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
                del self._spec[key]
            elif not isinstance(value, (list, tuple)):
                raise ValueError(f"Invalid value for key {key}: {value}")

    def match(self, field):
        if self._all:
            return True
        try:
            return all(field.get(key) in values for key, values in self._spec.items())
        except KeyError:
            return False


# ---------------------------------------------------------------------------
# Module-level constructors for the wrapper collection types.
#
# These are the free-function form of the ``Field`` / ``FieldList`` factory
# classmethods above, so callers can build fields without referencing the
# classes directly (and never earthkit-data directly).
# ---------------------------------------------------------------------------


def new_field_from_numpy(array: np.ndarray, *, template: Field, **metadata: Any) -> Field:
    """Create a new :class:`Field` from a numpy array (see :meth:`Field.from_numpy`)."""
    return Field.from_numpy(array, template=template, **metadata)


def new_field_from_latitudes_longitudes(template: Field, latitudes: np.ndarray, longitudes: np.ndarray) -> Field:
    """Create a new :class:`Field` from latitudes/longitudes (see :meth:`Field.from_latitudes_longitudes`)."""
    return Field.from_latitudes_longitudes(template, latitudes, longitudes)


def new_field_with_metadata(template: Field, **metadata: Any) -> Field:
    """Create a new :class:`Field` with updated metadata (see :meth:`Field.with_new_metadata`)."""
    return Field.with_new_metadata(template, **metadata)


def new_field_with_valid_datetime(template: Field, date: Any) -> Field:
    """Create a new :class:`Field` with a valid datetime (see :meth:`Field.with_valid_datetime`)."""
    return Field.with_valid_datetime(template, date)


def new_fieldlist_from_list(fields: list[Field]) -> FieldList:
    """Create a :class:`FieldList` from a list of fields (see :meth:`FieldList.from_fields`)."""
    return FieldList.from_fields(fields)
