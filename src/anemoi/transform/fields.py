# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The wrapper ``Field`` / ``FieldList`` abstraction over earthkit-data.

This module owns everything that pertains to fields and fieldlists: the
wrapper classes themselves, their constructors (``FieldList.from_source``,
``FieldList.concat``, ``Field.from_numpy``, ...) and ``new_grib_output``.
Non-field earthkit-data utilities (dates, patterns, availability, temporary
files, ...) are imported from ``earthkit.data`` directly at their call
sites.
"""

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

from anemoi.transform.data import DataContainer

LOG = logging.getLogger(__name__)

# Sentinel returned by a Flavour when it has no value for a given metadata key.
MISSING_METADATA = object()

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
    """
    return _GribOutput(path)


def __getattr__(name: str) -> Any:
    """Lazily re-export heavier / optional earthkit-data internals.

    Kept out of the eager import block so that importing
    :mod:`anemoi.transform.fields` stays cheap; the underlying module is only
    imported on first access (PEP 562).
    """
    _lazy = {
        "XArrayFieldList": ("earthkit.data.readers.xarray.fieldlist", "XArrayFieldList"),
    }
    if name in _lazy:
        import importlib

        module_name, attr = _lazy[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Legacy (GRIB-style) metadata key names mapped to earthkit-data 1.0
# component paths, for building new fields via ``set()``.
_METADATA_KEY_MAPPING = {
    "valid_datetime": "time.valid_datetime",
    "base_datetime": "time.base_datetime",
    "step": "time.step",
    "param": "parameter.variable",
    "units": "parameter.units",
    "levtype": "vertical.level_type",
    "levelist": "vertical.level",
    "number": "ensemble.member",
}


def _unwrap_field(field: "Field | _EkdField") -> _EkdField:
    """Return the underlying earthkit field for either a wrapped or raw field."""
    return field._field if isinstance(field, Field) else field


def _unwrap_fieldlist(fieldlist: "FieldList | _EkdFieldList") -> _EkdFieldList:
    """Return the underlying earthkit fieldlist for either a wrapped or raw fieldlist."""
    return fieldlist._fieldlist if isinstance(fieldlist, FieldList) else fieldlist


def _unwrap_any(value: Any) -> Any:
    """Unwrap a value for consumption by earthkit-data, if it is a wrapped Field/FieldList.

    Some earthkit-data sources (e.g. ``forcings``) take a field or fieldlist as
    an argument to use as a template; they only know about raw earthkit-data
    types, so any wrapper must be stripped before the value reaches them.
    """
    if isinstance(value, FieldList):
        return _unwrap_fieldlist(value)
    if isinstance(value, Field):
        return _unwrap_field(value)
    return value


class Field:
    """A thin, transparent wrapper around an earthkit-data field.

    Attribute access that is not explicitly defined here is delegated to the
    underlying earthkit field, so component accessors such as ``parameter``,
    ``time``, ``geography``, ``vertical`` and ``ensemble`` remain available.
    """

    def __init__(self, field: _EkdField | None = None):
        """Wrap a raw earthkit-data field.

        Parameters
        ----------
        field : earthkit.data.Field, optional
            The underlying earthkit-data field.
        """
        self._field = field

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying earthkit field.

        Only called when normal attribute lookup fails, so the wrapper's
        own methods and properties take precedence.
        """
        if name == "_field":
            raise AttributeError(name)
        return getattr(self._field, name)

    @property
    def name(self) -> str:
        """The name of the field (the ``labels.name`` label).

        The name is attached by a naming scheme (see
        :mod:`anemoi.transform.naming`) or explicitly with
        :meth:`with_name`; accessing it on a field that has not been
        named is an error.
        """
        name = self._field.get("labels.name", default=None)
        if name is None:
            raise ValueError(f"Field has no name (labels.name not set): {self._field}")
        return name

    @property
    def valid_datetime(self) -> datetime.datetime:
        """The valid datetime of the field (``time.valid_datetime``), timezone-naive."""
        value = self._field.get("time.valid_datetime", default=None)
        if value is None:
            raise ValueError(f"Field has no valid datetime: {self._field}")
        assert isinstance(value, datetime.datetime), f"Expected a datetime, got {type(value)}: {value!r}"
        assert value.tzinfo is None, f"Expected a timezone-naive datetime, got {value!r}"
        return value

    @property
    def param(self) -> str:
        """The variable name of the field (``parameter.variable``)."""
        value = self._field.get("parameter.variable", default=None)
        if value is None:
            raise ValueError(f"Field has no variable name: {self._field}")
        return value

    @property
    def number(self) -> int:
        """The ensemble member of the field (``metadata.number``).

        Fields that carry no ensemble information are member 0, following
        the convention used throughout the pipelines.
        """
        return self._field.get("metadata.number", default=None) or 0

    # === forwarded methods for Field class

    def set(self, *args, **kwargs) -> "Field":
        """Return a new :class:`Field` with the given components overridden.

        Thin forward of :meth:`earthkit.data.Field.set` that re-wraps the
        result.
        """
        return Field(self._field.set(*args, **kwargs))

    def get(self, *args, **kwargs) -> Any:
        """Get a metadata value by component path (e.g. ``"vertical.level"``).

        Thin forward of :meth:`earthkit.data.Field.get`.
        """
        return self._field.get(*args, **kwargs)

    def to_numpy(self, *args, **kwargs) -> Any:
        """Return the field values as a numpy array.

        Thin forward of :meth:`earthkit.data.Field.to_numpy`.
        """
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
        unknown_keys = set(metadata.keys()) - set(_METADATA_KEY_MAPPING.keys())
        if unknown_keys:
            raise ValueError(
                f"Unknown metadata keys: {unknown_keys}. Allowed keys are: {set(_METADATA_KEY_MAPPING.keys())}"
            )

        # map metadata keys to new locations
        mapped_metadata = {_METADATA_KEY_MAPPING[key]: value for key, value in metadata.items()}
        return cls(_unwrap_field(template).set(**mapped_metadata))

    @classmethod
    def with_name(cls, field: "Field", name: str) -> "Field":
        """Create a new field carrying ``name`` as its ``labels.name`` label.

        Parameters
        ----------
        field : Field
            The field to name.
        name : str
            The name to attach.

        Returns
        -------
        Field
            The new named field.
        """
        return cls(_unwrap_field(field).set(**{"labels.name": name}))

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
        """Create a new field with metadata overrides computed from a flavour.

        The flavour's rules are evaluated eagerly against the field and the
        resulting values are baked into a new field via ``set()``, so they
        survive operations that drop wrapper objects (``sel``, GRIB encoding,
        rebuilding fieldlists from raw fields, ...).

        Parameters
        ----------
        field : Field
            The field to which the flavour is applied.
        flavour : Flavour
            The flavour used to resolve metadata keys. Must expose its target
            keys via a ``rules`` mapping (e.g. ``RuleBasedFlavour``).

        Returns
        -------
        Field
            The new field with the flavour's metadata applied.
        """
        raw = _unwrap_field(field)
        wrapped = field if isinstance(field, Field) else cls(raw)
        rules = getattr(flavour, "rules", None)
        if rules is None:
            raise NotImplementedError(f"Cannot apply a flavour of type {type(flavour).__name__}: no 'rules' mapping")

        overrides = {}
        for key in rules:
            value = flavour(key, wrapped)
            if value is not MISSING_METADATA:
                overrides[_METADATA_KEY_MAPPING.get(key, f"metadata.{key}")] = value

        if not overrides:
            return wrapped
        return cls(raw.set(**overrides))


class FieldList(DataContainer):
    """A thin, transparent wrapper around an earthkit-data fieldlist.

    Iterating or indexing a :class:`FieldList` yields :class:`Field` objects.
    Attribute access that is not explicitly defined here is delegated to the
    underlying earthkit fieldlist.
    """

    def __init__(self, fieldlist: _EkdFieldList | None = None):
        """Wrap a raw earthkit-data fieldlist.

        Parameters
        ----------
        fieldlist : earthkit.data.FieldList, optional
            The underlying earthkit-data fieldlist. When omitted, the
            fieldlist is empty.
        """
        self._fieldlist = fieldlist if fieldlist is not None else ekd.create_fieldlist()
        self._fields: list[Field] | None = None

    @property
    def _underlying(self) -> _EkdFieldList:
        """The underlying earthkit-data fieldlist (see :class:`DataContainer`)."""
        return self._fieldlist

    @property
    def _wrapped(self) -> list[Field]:
        """The fields as wrapped :class:`Field` objects (built lazily, cached)."""
        if self._fields is None:
            self._fields = [f if isinstance(f, Field) else Field(f) for f in self._fieldlist]
        return self._fields

    @classmethod
    def from_fields(cls, fields: list[Field]) -> "FieldList":
        """Create a :class:`FieldList` from a list of fields.

        Parameters
        ----------
        fields : list of Field
            The fields (wrapped or raw) to build the fieldlist from. The
            identity of wrapped fields is preserved: indexing the result
            returns the same objects.

        Returns
        -------
        FieldList
            The new fieldlist.
        """
        fields = [f if isinstance(f, Field) else Field(f) for f in fields]
        result = cls(ekd.create_fieldlist([f._field for f in fields]))
        # Preserve the identity of the provided fields.
        result._fields = fields
        return result

    @classmethod
    def from_dicts(cls, dicts: list[dict]) -> "FieldList":
        """Create a :class:`FieldList` from a list of dictionaries.

        Parameters
        ----------
        dicts : list of dict
            One dictionary per field, with the field components (``values``,
            ``geography``, ``time``, ...), as accepted by the earthkit-data
            ``list-of-dicts`` source.

        Returns
        -------
        FieldList
            The new fieldlist.
        """
        return cls(ekd.from_source("list-of-dicts", dicts).to_fieldlist())

    @classmethod
    def from_source(cls, name: str, *args, **kwargs) -> "FieldList":
        """Create a :class:`FieldList` from an earthkit-data source.

        Wrapped ``Field``/``FieldList`` arguments are unwrapped before they
        reach earthkit-data (some sources, e.g. ``forcings``, take a field
        or fieldlist as template).

        Parameters
        ----------
        name : str
            The name of the earthkit-data source (``"file"``, ``"mars"``,
            ``"forcings"``, ...).
        *args : Any
            Positional arguments for the source.
        **kwargs : Any
            Keyword arguments for the source.

        Returns
        -------
        FieldList
            The new fieldlist.
        """
        args = tuple(_unwrap_any(a) for a in args)
        kwargs = {k: _unwrap_any(v) for k, v in kwargs.items()}
        result = ekd.from_source(name, *args, **kwargs)
        # Calling .to_fieldlist() on a value that is already a FieldList
        # (e.g. when a source is mocked in tests) can corrupt some
        # lazily-computed fields (observed with the "forcings" source), so
        # only convert when the source hasn't already produced one.
        if not isinstance(result, _EkdFieldList):
            result = result.to_fieldlist()
        return cls(result)

    @classmethod
    def from_file(cls, path: str, *, keep: Any = None) -> "FieldList":
        """Create a :class:`FieldList` from a file.

        Parameters
        ----------
        path : str
            The path of the file to read.
        keep : Any, optional
            An object to keep alive for as long as the fields (or any field
            derived from them via ``set()``) are alive — typically a
            temporary-file handle whose deletion would invalidate the
            fields' underlying GRIB handles.

        Returns
        -------
        FieldList
            The new fieldlist.
        """
        result = cls.from_source("file", path)
        if keep is not None:
            # Pin on the data component: it is shared with every field derived
            # from these via ``set()``, whereas the field object itself is not.
            for field in result._fieldlist:
                field._components["data"]._anemoi_keep = keep
        return result

    @classmethod
    def concat(cls, *args: "FieldList") -> "FieldList":
        """Concatenate multiple fieldlists into a single :class:`FieldList`.

        In earthkit-data 1.0 ``fieldlist + fieldlist`` is element-wise
        arithmetic; this is the way to concatenate collections.

        Parameters
        ----------
        *args : FieldList
            The fieldlists (wrapped or raw) to concatenate.

        Returns
        -------
        FieldList
            The concatenated fieldlist.
        """
        result = ekd.concat(*[_unwrap_fieldlist(arg) for arg in args])
        # See the comment in from_source(): avoid a redundant .to_fieldlist()
        # call, which can corrupt already-materialised fields.
        if not isinstance(result, _EkdFieldList):
            result = result.to_fieldlist()
        return cls(result)

    def to_fieldlist(self) -> "FieldList":
        """Return self.

        Without this, ``.to_fieldlist()`` (a common idiom for normalising an
        earthkit-data source into a fieldlist) would fall through to
        ``__getattr__`` and delegate to the underlying earthkit fieldlist,
        silently unwrapping this object back into a raw earthkit type.
        """
        return self

    def sel(self, *args, **kwargs) -> "FieldList":
        """Select a subset of the fields, returning a new :class:`FieldList`."""
        return FieldList(self._fieldlist.sel(*args, **kwargs))

    def order_by(self, *args, **kwargs) -> "FieldList":
        """Order the fields, returning a new :class:`FieldList`."""
        return FieldList(self._fieldlist.order_by(*args, **kwargs))

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self._fieldlist)

    def __getitem__(self, index: int) -> Field:
        """Return the field at ``index``, as a wrapped :class:`Field`."""
        return self._wrapped[index]

    def __iter__(self):
        """Iterate over the fields, as wrapped :class:`Field` objects."""
        return iter(self._wrapped)


class Flavour(ABC):
    """Base class for flavours: per-field metadata overrides.

    A flavour resolves metadata keys for a field (e.g. from a set of rules)
    and is applied with :meth:`Field.flavoured`.
    """

    @abstractmethod
    def __call__(self, key: str, field: Field) -> Any:
        """Return the flavour's value for ``key`` on ``field``.

        Parameters
        ----------
        key : str
            The metadata key being resolved.
        field : Field
            The field whose metadata is being resolved.

        Returns
        -------
        Any
            The value to use, or :data:`MISSING_METADATA` when the flavour
            has no value for this key.
        """
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
