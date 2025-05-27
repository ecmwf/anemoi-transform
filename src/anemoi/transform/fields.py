# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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


def new_fieldlist_from_list(fields: List[Any]) -> SimpleFieldList:
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


class WrappedField:
    """A wrapper around an earthkit-data field object.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    """

    def __init__(self, field: Any) -> None:
        self._field = field

    def __getattr__(self, name: str) -> Any:
        """Custom attribute access method for the WrappedField class.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            The value of the attribute from the underlying _field object.

        Raises
        ------
        AttributeError
            If the attribute name is "clone" or "copy".
        """
        if name in (
            "clone",
            "copy",
        ):
            raise AttributeError(f"{self}: forwarding of `{name}` is not supported")

        if name not in (
            "mars_area",
            "mars_grid",
            "to_numpy",
            "metadata",
            "shape",
            "grid_points",
            "handle",
        ):
            LOG.warning(f"{self}: forwarding `{name}`")

        return getattr(self._field, name)

    def __repr__(self) -> str:
        """Return the string representation of the field.

        Returns
        -------
        str
            The string representation of the `_field` attribute.
        """
        return f"{self.__class__.__name__ }({repr(self._field)}, {self._repr_specific()})"

    def _repr_specific(self) -> str:
        """Return a string representation of the specific field type.

        Returns
        -------
        str
            The string representation of the specific field type.
        """
        return f"(No specific representation for {self.__class__.__name__})"

    def clone(self, **kwargs: Any) -> "NewClonedField":
        """Clone the field with new metadata.

        Parameters
        ----------
        **kwargs : Any
            The new metadata for the cloned field.

        Returns
        -------
        NewClonedField
            The cloned field with the provided metadata.
        """
        return NewClonedField(self, **kwargs)

    def __iter__(self) -> Any:
        """Return an iterator over the field.

        Returns
        -------
        Any
            An iterator over the `_field` attribute.
        """
        raise NotImplementedError(f"{self}: iterating is not supported")


class NewDataField(WrappedField):
    """Change the data of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    data : np.ndarray
        The new data for the field.
    """

    def __init__(self, field: Any, data: np.ndarray) -> None:
        super().__init__(field)
        self._data = data
        self.shape = data.shape

    @property
    def values(self) -> np.ndarray:
        """Get the values of the field."""
        return self.to_numpy(flatten=True)

    def to_numpy(self, flatten: bool = False, dtype: Optional[type] = None, index: Optional[Any] = None) -> np.ndarray:
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
        data = self._data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        if index is not None:
            data = data[index]
        return data

    def _repr_specific(self) -> str:
        return f"(shape={self._data.shape})"


class GeoMetadata(Geography):
    """A wrapper around an earthkit-data Geography object.

    Parameters
    ----------
    owner : Any
        The owner of the geography data.
    """

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def shape(self) -> Tuple[int, ...]:
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

    def mars_area(self) -> List[float]:
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

    def latitudes(self, dtype: Optional[type] = None) -> np.ndarray:
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

    def longitudes(self, dtype: Optional[type] = None) -> np.ndarray:
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

    def x(self, dtype: Optional[type] = None) -> None:
        """Get the x-coordinates of the geography data."""
        raise NotImplementedError()

    def y(self, dtype: Optional[type] = None) -> None:
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


class NewLatLonField(WrappedField):
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

    def __init__(self, field: Any, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        super().__init__(field)
        self._latitudes = latitudes
        self._longitudes = longitudes

    def grid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the grid points of the field.

        Returns
        -------
        tuple
            The latitudes and longitudes of the field.
        """
        return self._latitudes, self._longitudes

    def to_latlon(self, flatten: bool = True) -> Dict[str, np.ndarray]:
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
        return dict(lat=self._latitudes, lon=self._longitudes)

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
        metadata = self._field.metadata(*args, **kwargs)
        if hasattr(metadata, "geography"):
            metadata.geography = GeoMetadata(self)

        return metadata


class NewGridField(WrappedField):
    """Change the grid of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    grid: Grid
        The new grid for the field.
    """

    def __init__(self, field: Any, grid: Grid) -> None:
        super().__init__(field)
        self._grid = grid

    def grid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the grid points of the field.

        Returns
        -------
        tuple
            The latitudes and longitudes of the field.
        """
        return self._grid.latlon()

    def to_latlon(self, flatten: bool = True) -> Dict[str, np.ndarray]:
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
        coords = self._grid.latlon()
        return dict(lat=coords[0], lon=coords[1])

    def __repr__(self) -> str:
        """Get the string representation of the field.

        Returns
        -------
        str
            The string representation of the field.
        """
        return f"NewGridField({self._field}, {self._grid})"

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
        metadata = self._field.metadata(*args, **kwargs)
        if hasattr(metadata, "geography"):
            metadata.geography = GeoMetadata(self)

        return metadata

    @property
    def _latitudes(self) -> np.ndarray:
        """Get the latitudes of the field."""
        return self._grid.latlon()[0]

    @property
    def _longitudes(self) -> np.ndarray:
        """Get the longitudes of the field."""
        return self._grid.latlon()[1]


class _NewMetadataField(WrappedField, ABC):
    """Change the metadata of a field."""

    def __init__(self, field: Any) -> None:
        super().__init__(field)

    @abstractmethod
    def mapping(self, key: str, field: ekd.Field) -> Any: ...

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

        if len(args) == 0 and len(kwargs) == 0:

            class MD:

                geography = this._field.metadata().geography

                def get(self, key, default=None):

                    value = this.mapping(key, this._field)
                    if value is not MISSING_METADATA:
                        return value

                    return this._field.metadata().get(key, default)

                def keys(self):
                    return this._field.metadata().keys()

            return MD()

        if kwargs.get("namespace"):
            assert len(args) == 0, (args, kwargs)
            mars = self._field.metadata(**kwargs).copy()
            for k in list(mars.keys()):
                m = self.mapping(k, self._field)
                if m is not MISSING_METADATA:
                    mars[k] = m
            return mars

        def _val(a):
            value = self.mapping(a, self._field)
            if value is MISSING_METADATA:
                return self._field.metadata(a, **kwargs)

            if callable(value):
                return value(self, a, self._field.metadata())

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

    def __init__(self, field: Any, **kwargs: Any) -> None:
        super().__init__(field)
        self.kwargs = kwargs

    def mapping(self, key: str, field: ekd.Field) -> Any:
        return self.kwargs.get(key, MISSING_METADATA)

    def _repr_specific(self):
        return f"(metadata={self.kwargs})"


class NewFlavouredField(_NewMetadataField):
    def __init__(self, field: Any, flavour: Flavour) -> None:
        super().__init__(field)
        self.flavour = flavour

    def mapping(self, key: str, field: ekd.Field) -> Any:
        return self.flavour(key, field)


class NewValidDateTimeField(NewMetadataField):
    """Change the valid_datetime of a field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    valid_datetime : Any
        The new valid_datetime for the field.
    """

    def __init__(self, field: Any, valid_datetime: Any) -> None:
        date = int(valid_datetime.date().strftime("%Y%m%d"))
        assert valid_datetime.time().minute == 0, valid_datetime
        time = valid_datetime.time().hour

        self.valid_datetime = valid_datetime

        super().__init__(field, date=date, time=time, step=0, valid_datetime=valid_datetime.isoformat())


class NewClonedField(WrappedField):
    """Wrapper around a field object that clones the field.

    Parameters
    ----------
    field : Any
        The field object to wrap.
    **metadata : Any
        The new metadata for the cloned field.
    """

    def __init__(self, field: Any, **metadata: Any) -> None:
        super().__init__(field)
        self._metadata = metadata

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Get the metadata of the cloned field.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The metadata of the cloned field.
        """
        if len(args) == 1:
            if args[0] in self._metadata:
                if callable(self._metadata[args[0]]):
                    proc = self._metadata[args[0]]
                    self._metadata[args[0]] = proc(self._field, args[0], self._field.metadata())

            if args[0] in self._metadata:
                return self._metadata[args[0]]

        return self._field.metadata(*args, **kwargs)

    def _repr_specific(self):
        return f"(metadata={self._metadata})"


def new_field_from_numpy(array: np.ndarray, *, template: WrappedField, **metadata: Any) -> NewMetadataField:
    """Create a new field from a numpy array.

    Parameters
    ----------
    array : np.ndarray
        The data for the new field.
    template : WrappedField
        The template field to use.
    **metadata : Any
        Additional metadata for the new field.

    Returns
    -------
    NewMetadataField
        The new field with the provided data and metadata.
    """
    return NewMetadataField(NewDataField(template, array), **metadata)


def new_field_with_valid_datetime(template: WrappedField, date: Any) -> NewValidDateTimeField:
    """Create a new field with a valid datetime.

    Parameters
    ----------
    template : WrappedField
        The template field to use.
    date : Any
        The valid datetime for the new field.

    Returns
    -------
    NewValidDateTimeField
        The new field with the provided valid datetime.
    """
    return NewValidDateTimeField(template, date)


def new_field_with_metadata(template: WrappedField, **metadata: Any) -> NewMetadataField:
    """Create a new field with metadata.

    Parameters
    ----------
    template : WrappedField
        The template field to use.
    **metadata : Any
        The metadata for the new field.

    Returns
    -------
    NewMetadataField
        The new field with the provided metadata.
    """
    return NewMetadataField(template, **metadata)


def new_field_from_latitudes_longitudes(
    template: WrappedField, latitudes: np.ndarray, longitudes: np.ndarray
) -> NewGridField:
    """Create a new field from latitudes and longitudes.

    Parameters
    ----------
    template : WrappedField
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
    return NewLatLonField(template, latitudes, longitudes)


def new_field_from_grid(
    template: WrappedField,
    grid: Grid,
) -> NewGridField:
    """Create a new field from a grid.

    Parameters
    ----------
    template : WrappedField
        The template field to use.
    grid : Grid
        The grid for the new field.

    Returns
    -------
    NewGridField
        The new field with the provided grid.
    """
    return NewGridField(template, grid)


def new_flavoured_field(field: Any, flavour: Flavour) -> NewFlavouredField:
    """Create a new field with a flavour."""
    return NewFlavouredField(field, flavour)


class FieldSelection:
    """A class for specifying which fields to process."""

    ALLOWED_KEYS = {"param", "levelist"}

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
            return all(field.metadata(key) in values for key, values in self._spec.items())
        except KeyError:
            return False
