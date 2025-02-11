# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
from earthkit.data.core.geography import Geography
from earthkit.data.indexing.fieldlist import FieldArray

LOG = logging.getLogger(__name__)


def new_fieldlist_from_list(fields):
    return FieldArray(fields)


def new_empty_fieldlist():
    return FieldArray([])


class WrappedField:
    """A wrapper around a earthkit-data field object."""

    def __init__(self, field):
        self._field = field

    def __getattr__(self, name):
        if name in (
            "clone",
            "copy",
        ):
            raise AttributeError(f"NewField: forwarding of `{name}` is not supported")

        if name not in (
            "mars_area",
            "mars_grid",
            "to_numpy",
            "metadata",
            "shape",
        ):
            LOG.warning(f"NewField: forwarding `{name}`")

        return getattr(self._field, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__ }({repr(self._field)})"

    def clone(self, **kwargs):
        return NewClonedField(self, **kwargs)


class NewDataField(WrappedField):
    """Change the data of a field."""

    def __init__(self, field, data):
        super().__init__(field)
        self._data = data
        self.shape = data.shape

    def to_numpy(self, flatten=False, dtype=None, index=None):
        data = self._data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        if index is not None:
            data = data[index]
        return data


class GeoMetadata(Geography):
    """A wrapper around a earthkit-data Geography object."""

    def __init__(self, owner):
        self.owner = owner

    def shape(self):
        return tuple([len(self.owner._latitudes)])

    def resolution(self):
        return "unknown"

    def mars_area(self):
        return [
            np.amax(self.owner._latitudes),
            np.amin(self.owner._longitudes),
            np.amin(self.owner._latitudes),
            np.amax(self.owner._longitudes),
        ]

    def mars_grid(self):
        return None

    def latitudes(self, dtype=None):
        if dtype is None:
            return self.owner._latitudes
        return self.owner._latitudes.astype(dtype)

    def longitudes(self, dtype=None):
        if dtype is None:
            return self.owner._longitudes
        return self.owner._longitudes.astype(dtype)

    def x(self, dtype=None):
        raise NotImplementedError()

    def y(self, dtype=None):
        raise NotImplementedError()

    def _unique_grid_id(self):
        raise NotImplementedError()

    def projection(self):
        return None

    def bounding_box(self):
        raise NotImplementedError()

    def gridspec(self):
        raise NotImplementedError()


class NewGridField(WrappedField):
    """Change the grid of a field."""

    def __init__(self, field, latitudes, longitudes):
        super().__init__(field)
        self._latitudes = latitudes
        self._longitudes = longitudes

    def grid_points(self):
        return self._latitudes, self._longitudes

    def to_latlon(self, flatten=True):
        assert flatten
        return dict(lat=self._latitudes, lon=self._longitudes)

    def __repr__(self):
        return f"NewGridField({len(self._latitudes), self._field})"

    def metadata(self, *args, **kwargs):

        metadata = self._field.metadata(*args, **kwargs)
        if hasattr(metadata, "geography"):
            metadata.geography = GeoMetadata(self)

        return metadata


class NewMetadataField(WrappedField):
    """Change the metadata of a field."""

    def __init__(self, field, **kwargs):
        super().__init__(field)
        self._metadata = kwargs

    def metadata(self, *args, **kwargs):

        this = self

        if len(args) == 0 and len(kwargs) == 0:

            class MD:

                def get(self, key, default=None):
                    if key in this._metadata:
                        return this._metadata[key]

                    return this._field.metadata().get(key, default)

            return MD()

        if kwargs.get("namespace"):
            assert len(args) == 0, (args, kwargs)
            mars = self._field.metadata(**kwargs).copy()
            for k in list(mars.keys()):
                if k in self._metadata:
                    mars[k] = self._metadata[k]
            return mars

        if len(args) == 1 and args[0] in self._metadata:
            value = self._metadata[args[0]]
            if callable(value):
                return value(self, args[0], self._field.metadata())
            return value

        return self._field.metadata(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__ }({repr(self._field)},{self._metadata})"


class NewValidDateTimeField(NewMetadataField):
    """Change the valid_datetime of a field."""

    def __init__(self, field, valid_datetime):
        date = int(valid_datetime.date().strftime("%Y%m%d"))
        assert valid_datetime.time().minute == 0, valid_datetime
        time = valid_datetime.time().hour

        self.valid_datetime = valid_datetime

        super().__init__(field, date=date, time=time, step=0, valid_datetime=valid_datetime.isoformat())


class NewClonedField(WrappedField):
    """Wrapper around a field object that clones the field."""

    def __init__(self, field, **metadata):
        super().__init__(field)
        self._metadata = metadata

    def metadata(self, *args, **kwargs):
        if len(args) == 1:
            if args[0] in self._metadata:
                if callable(self._metadata[args[0]]):
                    proc = self._metadata[args[0]]
                    self._metadata[args[0]] = proc(self._field, args[0], self._field.metadata())

            if args[0] in self._metadata:
                return self._metadata[args[0]]

        return self._field.metadata(*args, **kwargs)


def new_field_from_numpy(array, *, template=None, **metadata):
    return NewMetadataField(NewDataField(template, array), **metadata)


def new_field_with_valid_datetime(template, date):
    return NewValidDateTimeField(template, date)


def new_field_with_metadata(template, **metadata):
    return NewMetadataField(template, **metadata)


def new_field_from_latitudes_longitudes(template, latitudes, longitudes):
    return NewGridField(template, latitudes, longitudes)
