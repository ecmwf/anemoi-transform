# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import earthkit.data as ekd
import numpy as np
from earthkit.data.core.geography import Geography
from earthkit.data.indexing.fieldlist import SimpleFieldList

from anemoi.transform.grids import Grid

LOG = logging.getLogger(__name__)

MISSING_METADATA = object()


class Flavour(ABC):

    @abstractmethod
    def __call__(self, key: str, field: ekd.Field) -> Any:
        """Called during field metadata lookup, so it can be modified"""
        pass


def new_fieldlist_from_list(fields: list[Any]) -> SimpleFieldList:
    """Create a new SimpleFieldList from a list of fields.

    Parameters
    ----------
    fields : list
        List of fields to include in the FieldArray.

    Returns
    -------
    SimpleFieldList
        A new SimpleFieldList containing the provided fields.
    """
    return SimpleFieldList(fields)


def new_empty_fieldlist() -> SimpleFieldList:
    """Create a new empty SimpleFieldList.

    Returns
    -------
    SimpleFieldList
        A new empty SimpleFieldList.
    """
    return SimpleFieldList([])


class _Wrapper:

    def clone(self, *args: Any, **kwargs: Any) -> Any:
        assert not args
        return new_field_with_metadata(self, **kwargs)

    def copy(self) -> Any:
        """Create a copy of the wrapped field."""
        assert False, f"Not implemented {type(self)}"


class NewDataField(_Wrapper):
    """Change the data of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    data : np.ndarray
        The new data for the field.
    """

    @property
    def values(self) -> np.ndarray:
        """Get the values of the field."""
        return self.to_numpy(flatten=True)

    def to_numpy(self, flatten: bool = False, dtype: type | None = None, index: Any | None = None) -> np.ndarray:
        """Convert the field data to a numpy array.

        Parameters
        ----------
        flatten : bool, optional
            Whether to flatten the array, by default False.
        dtype : type, optional
            The desired data type of the array, by default None.
        index : Any, optional
            The index to apply to the array, by default None.

        Returns
        -------
        np.ndarray
            The field data as a numpy array.
        """

        data = self._wrapped_array
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        if index is not None:
            data = data[index]

        data.flags.writeable = False
        return data

    def __getstate__(self) -> dict[str, Any]:
        state = []
        try:
            state = super().__getstate__()
        except AttributeError:
            pass
        state["_wrapped_array"] = self._wrapped_array
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            super().__setstate__(state)
        except AttributeError:
            pass
        self._wrapped_array = state["_wrapped_array"]

    @classmethod
    def adjust_clone(cls, original: Any, clone: Any) -> Any:
        return new_field_from_numpy(original._wrapped_array, field=clone)


class GeoMetadata(Geography):
    """A wrapper around an earthkit-data Geography object.

    Parameters
    ----------
    owner : Any
        The owner of the geography data.
    """

    def shape(self) -> tuple[int, ...]:
        """Get the shape of the geography data.

        Returns
        -------
        tuple
            The shape of the geography data.
        """
        return tuple([len(self.owner._latitudes)])

    def resolution(self) -> str:
        """Get the resolution of the geography data.

        Returns
        -------
        str
            The resolution of the geography data.
        """
        return "unknown"

    def mars_area(self) -> list[float]:
        """Get the MARS area of the geography data.

        Returns
        -------
        list
            The MARS area of the geography data.
        """
        return [
            np.amax(self.owner._latitudes),
            np.amin(self.owner._longitudes),
            np.amin(self.owner._latitudes),
            np.amax(self.owner._longitudes),
        ]

    def mars_grid(self) -> None:
        """Get the MARS grid of the geography data."""
        return None

    def latitudes(self, dtype: type | None = None) -> np.ndarray:
        """Get the latitudes of the geography data.

        Parameters
        ----------
        dtype : type, optional
            The desired data type of the array, by default None.

        Returns
        -------
        np.ndarray
            The latitudes of the geography data.
        """
        if dtype is None:
            return self.owner._latitudes
        return self.owner._latitudes.astype(dtype)

    def longitudes(self, dtype: type | None = None) -> np.ndarray:
        """Get the longitudes of the geography data.

        Parameters
        ----------
        dtype : type, optional
            The desired data type of the array, by default None.

        Returns
        -------
        np.ndarray
            The longitudes of the geography data.
        """
        if dtype is None:
            return self.owner._longitudes
        return self.owner._longitudes.astype(dtype)

    def x(self, dtype: type | None = None) -> None:
        """Get the x-coordinates of the geography data."""
        raise NotImplementedError()

    def y(self, dtype: type | None = None) -> None:
        """Get the y-coordinates of the geography data."""
        raise NotImplementedError()

    def _unique_grid_id(self) -> None:
        """Get the unique grid ID of the geography data."""
        raise NotImplementedError()

    def projection(self) -> None:
        """Get the projection of the geography data."""
        return None

    def bounding_box(self) -> None:
        """Get the bounding box of the geography data."""
        raise NotImplementedError()

    def gridspec(self) -> None:
        """Get the grid specification of the geography data."""
        raise NotImplementedError()


class NewLatLonField(_Wrapper):
    """Change the latitudes and longitudes of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    latitudes : np.ndarray
        The new latitudes for the field.
    longitudes : np.ndarray
        The new longitudes for the field.
    """

    def grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the grid points of the field.

        Returns
        -------
        tuple
            The latitudes and longitudes of the field.
        """
        return self._wrapped_latitudes, self._wrapped_longitudes

    def to_latlon(self, flatten: bool = True) -> dict[str, np.ndarray]:
        """Convert the grid points to latitude and longitude.

        Parameters
        ----------
        flatten : bool, optional
            Whether to flatten the arrays, by default True.

        Returns
        -------
        dict
            A dictionary containing the latitudes and longitudes.
        """
        assert flatten
        return dict(lat=self._wrapped_latitudes, lon=self._wrapped_longitudes)

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Get the metadata of the field.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The metadata of the field.
        """
        metadata = super().metadata(*args, **kwargs)
        if hasattr(metadata, "geography"):
            metadata.geography = GeoMetadata(self)

        return metadata

    @property
    def _latitudes(self) -> np.ndarray:
        """Get the latitudes of the field."""
        return self._wrapped_latitudes

    @property
    def _longitudes(self) -> np.ndarray:
        """Get the longitudes of the field."""
        return self._wrapped_longitudes

    def __getstate__(self) -> dict[str, Any]:
        state = []
        try:
            state = super().__getstate__()
        except AttributeError:
            pass
        state["_wrapped_latitudes"] = self._wrapped_latitudes
        state["_wrapped_longitudes"] = self._wrapped_longitudes
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            super().__setstate__(state)
        except AttributeError:
            pass
        self._wrapped_latitudes = state["_wrapped_latitudes"]
        self._wrapped_longitudes = state["_wrapped_longitudes"]


class NewGridField(_Wrapper):
    """Change the grid of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    grid: Grid
        The new grid for the field.
    """

    def grid_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the grid points of the field.

        Returns
        -------
        tuple
            The latitudes and longitudes of the field.
        """
        return self._wrapped_grid.latlon()

    def to_latlon(self, flatten: bool = True) -> dict[str, np.ndarray]:
        """Convert the grid points to latitude and longitude.

        Parameters
        ----------
        flatten : bool, optional
            Whether to flatten the arrays, by default True.

        Returns
        -------
        dict
            A dictionary containing the latitudes and longitudes.
        """
        assert flatten
        coords = self._wrapped_grid.latlon()
        return dict(lat=coords[0], lon=coords[1])

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Get the metadata of the field.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The metadata of the field.
        """
        metadata = super().metadata(*args, **kwargs)
        if hasattr(metadata, "geography"):
            metadata.geography = GeoMetadata(self)

        return metadata

    @property
    def _latitudes(self) -> np.ndarray:
        """Get the latitudes of the field."""
        return self._wrapped_grid.latlon()[0]

    @property
    def _longitudes(self) -> np.ndarray:
        """Get the longitudes of the field."""
        return self._wrapped_grid.latlon()[1]

    def __getstate__(self) -> dict[str, Any]:
        state = []
        try:
            state = super().__getstate__()
        except AttributeError:
            pass
        state["_wrapped_grid"] = self._wrapped_grid
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            super().__setstate__(state)
        except AttributeError:
            pass
        self._wrapped_grid = state["_wrapped_grid"]


class _NewMetadataField(_Wrapper):
    """Change the metadata of a field."""

    def mapping(self, key: str, field: ekd.Field) -> Any:
        # We cannot use a ABC with dynamic class creation
        raise NotImplementedError()

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Get the metadata of the field.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The metadata of the field.
        """
        this = self
        super_metadata = super().metadata

        if len(args) == 0 and len(kwargs) == 0:

            class MD:

                geography = super_metadata().geography

                def get(self, key, default=None):
                    return this.get(key, default)

                def keys(self):
                    return this.keys()

            return MD()

        if kwargs.get("namespace"):
            assert len(args) == 0, (args, kwargs)
            mars = super_metadata(**kwargs).copy()
            for k in list(mars.keys()):
                m = self.mapping(k)
                if m is not MISSING_METADATA:
                    mars[k] = m
            return mars

        def _val(a):
            value = self.mapping(a)
            if value is MISSING_METADATA:
                return super_metadata(a, **kwargs)

            if callable(value):
                return value(self, a, super_metadata())

            return value

        result = [_val(a) for a in args]
        if len(result) == 1:
            return result[0]

        return tuple(result)


class NewMetadataField(_NewMetadataField):
    """Change the metadata of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    **kwargs : Any
        The new metadata for the field.
    """

    def mapping(self, key: str) -> Any:
        return self._wrapped_metadata.get(key, MISSING_METADATA)

    def keys(self) -> set[str]:
        return self._wrapped_keys | set(self._wrapped_metadata.keys())

    def get(self, key: str, default: Any = MISSING_METADATA) -> Any:
        value = self._wrapped_metadata.get(key, MISSING_METADATA)
        if value is not MISSING_METADATA:
            return value

        if default is not MISSING_METADATA:
            return self._wrapped_get(key, default=default)

        return self._wrapped_get(key)

    def __getstate__(self) -> dict[str, Any]:
        state = []
        try:
            state = super().__getstate__()
        except AttributeError:
            pass
        state["_wrapped_metadata"] = self._wrapped_metadata
        state["_wrapped_keys"] = self._wrapped_keys
        state["_wrapped_get"] = self._wrapped_get
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            super().__setstate__(state)
        except AttributeError:
            pass
        self._wrapped_metadata = state["_wrapped_metadata"]
        self._wrapped_keys = state["_wrapped_keys"]
        self._wrapped_get = state["_wrapped_get"]


class NewFlavouredField(_NewMetadataField):

    def mapping(self, key: str, field: ekd.Field) -> Any:
        return self._wrapped_flavour(key, field)

    def __getstate__(self) -> dict[str, Any]:
        state = []
        try:
            state = super().__getstate__()
        except AttributeError:
            pass
        state["_wrapped_flavour"] = self._wrapped_flavour
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            super().__setstate__(state)
        except AttributeError:
            pass
        self._wrapped_flavour = state["_wrapped_flavour"]

    def keys(self) -> set[str]:
        raise NotImplementedError()


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
    NewMetadataField
        The new field with the provided data and metadata.
    """
    f = copy.copy(template)  # Shallow copy

    if isinstance(f, NewDataField):
        f._wrapped_array = array
    else:
        f.__class__ = type(
            f"{template.__class__.__name__}NewDataField",
            (
                NewDataField,
                template.__class__,
            ),
            {"_wrapped_array": array},
        )

    if metadata:
        f = new_field_with_metadata(f, **metadata)

    return f


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

    f = copy.copy(template)  # Shallow copy

    if isinstance(f, NewMetadataField):
        f._wrapped_metadata = f._wrapped_metadata.copy()
        f._wrapped_metadata.update(metadata)
    else:
        f.__class__ = type(
            f"{template.__class__.__name__}NewMetadataField",
            (NewMetadataField, template.__class__),
            {
                "_wrapped_metadata": metadata,
                "_wrapped_keys": set(template.metadata().keys()),
                "_wrapped_get": template.metadata().get,
            },
        )

    return f


def new_field_with_valid_datetime(template: ekd.Field, valid_datetime: Any) -> ekd.Field:
    """Create a new field with a valid datetime.

    Parameters
    ----------
    template : ekd.Field
        The template field to use.
    valid_datetime : Any
        The valid datetime for the new field.

    Returns
    -------
    NewValidDateTimeField
        The new field with the provided valid datetime.
    """
    date = int(valid_datetime.date().strftime("%Y%m%d"))
    assert valid_datetime.time().minute == 0, valid_datetime
    time = valid_datetime.time().hour

    return new_field_with_metadata(template, date=date, time=time, step=0, valid_datetime=valid_datetime.isoformat())


def new_field_from_latitudes_longitudes(
    template: ekd.Field,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
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
    NewGridField
        The new field with the provided latitudes and longitudes.
    """
    f = copy.copy(template)  # Shallow copy

    if isinstance(f, NewLatLonField):
        f._wrapped_latitudes = latitudes
        f._wrapped_longitudes = longitudes
    else:
        f.__class__ = type(
            f"{template.__class__.__name__}NewLatLonField",
            (NewLatLonField, template.__class__),
            {"_wrapped_latitudes": latitudes, "_wrapped_longitudes": longitudes},
        )

    return f


def new_field_from_grid(template: ekd.Field, grid: Grid) -> ekd.Field:
    """Create a new field from a grid.

    Parameters
    ----------
    template : ekd.Field
        The template field to use.
    grid : Grid
        The grid for the new field.

    Returns
    -------
    NewGridField
        The new field with the provided grid.
    """

    f = copy.copy(template)  # Shallow copy

    if isinstance(f, NewGridField):
        f._wrapped_grid = grid
    else:
        f.__class__ = type(
            f"{template.__class__.__name__}NewGridField",
            (NewGridField, template.__class__),
            {"_wrapped_grid": grid},
        )

    return f


def new_flavoured_field(template: ekd.Field, flavour: Flavour) -> ekd.Field:

    f = copy.copy(template)  # Shallow copy

    if isinstance(f, NewFlavouredField):
        f._wrapped_flavour = flavour
    else:
        f.__class__ = type(
            f"{template.__class__.__name__}NewFlavouredField",
            (NewFlavouredField, template.__class__),
            {"_wrapped_flavour": flavour},
        )

    return f


class FieldSelection:
    """A class for specifying which fields to process."""

    ALLOWED_KEYS = {"param", "levelist"}

    def __init__(self, **kwargs: Any) -> None:
        self._spec = kwargs
        self._validate_spec()
        self._sanitise_spec()
        self._all = len(self._spec) == 0

    def _validate_spec(self) -> None:
        if not set(self._spec).issubset(self.ALLOWED_KEYS):
            raise ValueError(f"Invalid keys in spec: {tuple(self._spec)} - only {self.ALLOWED_KEYS} are allowed.")

    def _sanitise_spec(self) -> None:
        for key, value in list(self._spec.items()):
            if isinstance(value, (str, int, float, bool)):
                self._spec[key] = (value,)
            elif value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
                del self._spec[key]
            elif not isinstance(value, (list, tuple)):
                raise ValueError(f"Invalid value for key {key}: {value}")

    def match(self, field: Any) -> bool:
        if self._all:
            return True
        try:
            return all(field.metadata(key) in values for key, values in self._spec.items())
        except KeyError:
            return False
