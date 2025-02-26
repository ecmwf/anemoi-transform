# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from earthkit.data.core.geography import Geography
from earthkit.data.indexing.fieldlist import SimpleFieldList

LOG = logging.getLogger(__name__)


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
        """Custom attribute access method for the NewField class.

        This method intercepts attribute access and provides custom behavior for certain attributes.
        If the attribute name is "clone" or "copy", an AttributeError is raised indicating that forwarding
        of these attributes is not supported. For other attributes, a warning is logged if the attribute
        name is not in the predefined list of supported attributes.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute from the underlying _field object.

        Raises:
            AttributeError: If the attribute name is "clone" or "copy".
        """
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
        """Return the string representation of the field.

        This method returns the string representation of the `_field` attribute,
        which provides a human-readable description of the object.

        Returns:
            str: The string representation of the `_field` attribute.
        """
        return f"{self.__class__.__name__ }({repr(self._field)})"

    def clone(self, **kwargs):
        return NewClonedField(self, **kwargs)


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


class NewGridField(WrappedField):
    """Change the grid of a field.

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

    def __repr__(self) -> str:
        """Get the string representation of the field.

        Returns
        -------
        str
            The string representation of the field.
        """
        return f"NewGridField({len(self._latitudes), self._field})"

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


class NewMetadataField(WrappedField):
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
        self._metadata = kwargs

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
    return NewGridField(template, latitudes, longitudes)
