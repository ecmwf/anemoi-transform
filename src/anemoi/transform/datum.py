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

LOG = logging.getLogger(__name__)


class Datum(ABC):
    """Abstract base class for transparent wrappers around a backing container.

    A :class:`Datum` is a thin, transparent wrapper around some underlying data
    container (for example an earthkit-data field list or a pandas DataFrame).
    Concrete subclasses such as :class:`anemoi.transform.fields.FieldList` and
    :class:`anemoi.transform.frames.Frame` store that container and expose it
    through the :attr:`_underlying` property.

    Attribute access that is not explicitly defined on a subclass is delegated
    to the underlying container via :meth:`__getattr__`, so the container's own
    accessors and methods remain available on the wrapper. Subclasses must also
    implement the common factory class methods (:meth:`from_dicts`,
    :meth:`from_file`, :meth:`concat`) and the container protocol
    (:meth:`__len__`, :meth:`__getitem__`, :meth:`__iter__`).
    """

    @property
    @abstractmethod
    def _underlying(self) -> Any:
        """The underlying container being wrapped."""
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails.
        # Private attributes (including the wrapper's own bookkeeping) are never
        # delegated, which also avoids recursion while the wrapper is being
        # initialised (before its backing attribute exists).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._underlying, name)

    @classmethod
    @abstractmethod
    def from_dicts(cls, dicts: list[dict]) -> "Datum":
        """Create a Datum from a list of dictionaries."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(cls, path: str) -> "Datum":
        """Create a Datum from a file."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concat(cls, *args: "Datum") -> "Datum":
        """Concatenate multiple Datums into a single Datum."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
