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

from anemoi.transform.datum import Datum

LOG = logging.getLogger(__name__)

# Sentinel returned by a Flavour when it has no value for a given metadata key.
MISSING_METADATA = object()


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
