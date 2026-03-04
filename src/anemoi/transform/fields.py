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
from earthkit.data.field.component.geography import create_geography_from_dict

LOG = logging.getLogger(__name__)


class Flavour(ABC):
    @abstractmethod
    def __call__(self, key: str, field: ekd.Field) -> Any:
        """Called during field metadata lookup, so it can be modified"""
        pass


def new_fieldlist_from_list(fields: list[ekd.Field]) -> ekd.FieldList:
    """Create a new FieldList from a list of fields.

    Parameters
    ----------
    fields : list[ekd.Field]
        List of fields to include in the fieldlist.

    Returns
    -------
    ekd.FieldList
        A new FieldList containing the provided fields.
    """
    return ekd.create_fieldlist(fields)


def new_empty_fieldlist() -> ekd.FieldList:
    """Create a new empty SimpleFieldList.

    Returns
    -------
    SimpleFieldList
        A new empty SimpleFieldList.
    """
    return ekd.create_fieldlist()


def new_field_from_numpy(array: np.ndarray, *, template: ekd.Field, **metadata: Any) -> ekd.Field:
    """Create a new field from a numpy array.

    Parameters
    ----------
    array : np.ndarray
        The data for the new field.
    template : ekd.Field
        The template field to use.
    **metadata : Any
        Additional metadata for the new field.

    Returns
    -------
    ekd.Field
        The new field with the provided data and metadata.
    """
    new_data = template.set(**{"data.values": array})
    if not metadata:
        return new_data
    return new_field_with_metadata(new_data, **metadata)


def new_field_with_valid_datetime(template: ekd.Field, date: Any) -> ekd.Field:
    """Create a new field with a valid datetime (sets the step to 0)
    therefore updating the base_datetime as well.

    Parameters
    ----------
    template : ekd.Field
        The template field to use.
    date : Any
        The valid datetime for the new field.

    Returns
    -------
    ekd.Field
        The new field with the provided valid datetime.
    """
    time = template.time.set(valid_datetime=date, step=datetime.timedelta(hours=0))
    return template.set(time=time)


def new_field_with_metadata(template: ekd.Field, **metadata: Any) -> ekd.Field:
    """Create a new field with metadata.

    Parameters
    ----------
    template : ekd.Field
        The template field to use.
    **metadata : Any
        The metadata for the new field.

    Returns
    -------
    ekd.Field
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
    return template.set(**mapped_metadata)


def new_field_from_latitudes_longitudes(
    template: ekd.Field, latitudes: np.ndarray, longitudes: np.ndarray
) -> ekd.Field:
    """Create a new field from latitudes and longitudes.

    Parameters
    ----------
    template : ekd.Field
        The template field to use.
    latitudes : np.ndarray
        The latitudes for the new field.
    longitudes : np.ndarray
        The longitudes for the new field.

    Returns
    -------
    ekd.Field
        The new field with the provided latitudes and longitudes.
    """
    geography = create_geography_from_dict(
        {
            "latitudes": latitudes,
            "longitudes": longitudes,
            # should projTargetString be passed in?
            #            "projTargetString": template.geography.projection().to_proj_string(),
        }
    )
    return template.set(geography=geography)


def new_flavoured_field(field: ekd.Field, flavour: Flavour) -> ekd.Field:
    """Create a new field with a flavour."""
    raise NotImplementedError("Not implemented yet.")


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
