# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

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
        if name not in (
            "mars_area",
            "mars_grid",
            "to_numpy",
            "metadata",
        ):
            LOG.warning(f"NewField: forwarding `{name}`")
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)


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


class NewMetadataField(WrappedField):
    """Change the metadata of a field."""

    def __init__(self, field, **kwargs):
        super().__init__(field)
        self._metadata = kwargs

    def metadata(self, *args, **kwargs):

        if kwargs.get("namespace"):
            assert len(args) == 0, (args, kwargs)
            mars = self._field.metadata(**kwargs).copy()
            for k in list(mars.keys()):
                if k in self._metadata:
                    mars[k] = self._metadata[k]
            return mars

        if len(args) == 1 and args[0] in self._metadata:
            return self._metadata[args[0]]

        return self._field.metadata(*args, **kwargs)


class NewValidDateTimeField(NewMetadataField):
    """Change the valid_datetime of a field."""

    def __init__(self, field, valid_datetime):
        date = int(valid_datetime.date().strftime("%Y%m%d"))
        assert valid_datetime.time().minute == 0, valid_datetime.time()
        time = valid_datetime.time().hour

        self.valid_datetime = valid_datetime

        super().__init__(field, date=date, time=time, step=0, valid_datetime=valid_datetime.isoformat())


def new_field_from_numpy(array, *, template, **metadata):
    return NewMetadataField(NewDataField(template, array), **metadata)


def new_field_with_valid_datetime(template, date):
    return NewValidDateTimeField(template, date)
