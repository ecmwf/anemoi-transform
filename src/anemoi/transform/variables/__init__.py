# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Typed representation of meteorological variables.

Variables are collected when datasets are created (from the fields, see
:class:`~anemoi.transform.variables.from_field.VariableFromField`),
serialised to JSON in the dataset metadata, copied to checkpoint
metadata at training time, and deserialised again with
:meth:`Variable.from_dict` in training and inference.

The serialised form is self-describing: each variable dictionary may
carry a ``"schema"`` key naming the layout it was written with (see
``VARIABLE_SCHEMAS``). Dictionaries without a ``"schema"`` key are the
historical MARS-vocabulary layout, so every existing dataset and
checkpoint keeps deserialising. Versioning is per variable rather than
per collection because collections are recombined across datasets of
different vintages (join, select, rename, complement).

Each layout is modelled with pydantic in
:mod:`anemoi.transform.variables.schemas`; the class ``from_dict``
dispatches to validates the dictionary against its model
(``schema_model``), and ``as_dict`` dumps it, so both directions of the
JSON round-trip are validated.
"""

import importlib
import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

if TYPE_CHECKING:
    from datetime import timedelta

LOG = logging.getLogger(__name__)

# Serialisation schema name → implementing class, as "module.Class" paths
# (resolved lazily to avoid import cycles).  The `None` entry handles
# legacy dictionaries that predate the "schema" key.
VARIABLE_SCHEMAS: dict[str | None, str] = {
    None: "anemoi.transform.variables.from_dict.VariableFromDict",
    "variable/1": "anemoi.transform.variables.components.VariableFromComponents",
}


def register_variable_schema(schema: str, class_path: str) -> None:
    """Register a serialisation schema for :meth:`Variable.from_dict`.

    Parameters
    ----------
    schema : str
        The schema name, as stored under the ``"schema"`` key of the
        serialised variable (e.g. ``"variable/2"``).
    class_path : str
        Fully qualified ``module.Class`` path of the class implementing
        the schema. The class is instantiated as ``Class(name, data)``.
    """
    VARIABLE_SCHEMAS[schema] = class_path


class Variable(ABC):
    """Variable is a class that represents a variable during
    training and inference.
    """

    def __init__(self, name: str) -> None:
        """Parameters
        -------------
        name : str
            The name of the variable.
        """
        self.name: str = name

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Variable":
        """Create a Variable instance from a serialised dictionary.

        The concrete class is selected from the dictionary's ``"schema"``
        key via ``VARIABLE_SCHEMAS``; dictionaries without that key use
        the legacy MARS-vocabulary layout.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : Dict[str, Any]
            The serialised variable.

        Returns
        -------
        Variable
            The created Variable instance.

        Raises
        ------
        ValueError
            If the dictionary carries an unknown ``"schema"`` (e.g. it
            was written by a newer version of this package).
        """
        schema = data.get("schema")
        class_path = VARIABLE_SCHEMAS.get(schema)
        if class_path is None:
            raise ValueError(
                f"Variable {name!r}: unknown serialisation schema {schema!r}"
                f" (known: {sorted(k for k in VARIABLE_SCHEMAS if k is not None)})."
                " It may have been written by a newer version of anemoi-transform."
            )
        module_name, _, class_name = class_path.rpartition(".")
        klass = getattr(importlib.import_module(module_name), class_name)
        return klass(name, data)

    @classmethod
    def from_field(cls, name: str, field: Any) -> Any:
        """Create a Variable instance from a field.

        Parameters
        ----------
        name : str
            The name of the variable.
        field : Field
            The field describing the variable.

        Returns
        -------
        Any
            The created Variable instance.
        """
        from anemoi.transform.variables.from_field import VariableFromField

        return VariableFromField(name, field)

    @classmethod
    def from_earthkit(cls, name: str, field: Any) -> Any:
        """Deprecated alias of :meth:`from_field`.

        Parameters
        ----------
        name : str
            The name of the variable.
        field : Field
            The field describing the variable.

        Returns
        -------
        Any
            The created Variable instance.
        """
        import warnings

        warnings.warn(
            "'Variable.from_earthkit' is deprecated. Please use 'Variable.from_field' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_field(name, field)

    def __repr__(self) -> str:
        """Return a string representation of the Variable.

        Returns
        -------
        str
            The name of the variable.
        """
        return self.name

    def __hash__(self) -> int:
        """Return the hash of the Variable.

        Returns
        -------
        int
            The hash of the variable name.
        """
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """Check if two Variable instances are equal.

        Parameters
        ----------
        other : Any
            The other variable to compare with.

        Returns
        -------
        bool
            True if the variables are equal, False otherwise.
        """
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    @property
    @abstractmethod
    def _level_type(self) -> str | None:
        """The level type as a MARS-style abbreviation (``"sfc"``, ``"pl"``, ...), or None when unknown."""
        pass

    @property
    def is_pressure_level(self) -> bool | None:
        """Check if the variable is a pressure level (None when the level type is unknown)."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "pl"

    @property
    def is_model_level(self) -> bool | None:
        """Check if the variable is a model level (None when the level type is unknown)."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "ml"

    @property
    def is_surface_level(self) -> bool | None:
        """Check if the variable is on the surface (None when the level type is unknown)."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "sfc"

    @property
    @abstractmethod
    def level(self) -> Any:
        """Get the level of the variable."""
        pass

    @property
    @abstractmethod
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time."""
        pass

    @property
    def is_valid_over_a_period(self) -> bool:
        """Check if the variable is valid over a period (e.g. accumulated or averaged)."""
        return self.time_processing is not None

    @property
    def is_instantaneous(self) -> bool:
        """Check if the variable is instantaneous."""
        return not self.is_valid_over_a_period

    @property
    @abstractmethod
    def time_processing(self):
        """Get the time processing type of the variable."""
        pass

    @property
    @abstractmethod
    def period(self) -> Union["timedelta", None]:
        """Get the variable's period as a timedelta.
        For instantaneous variables, returns a timedelta of 0. For non-instantaneous variables, returns `None` if this information is missing.
        """
        pass

    @property
    def is_accumulation(self) -> bool:
        """Check if the variable is an accumulation."""
        return self.time_processing == "accumulation"

    @property
    def param(self) -> str:
        """Get the parameter name of the variable."""
        return self.name

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Serialise the variable to a JSON-compatible dictionary.

        The result round-trips through :meth:`from_dict`: it either
        carries a ``"schema"`` key naming its layout or uses the legacy
        MARS-vocabulary layout.

        Returns
        -------
        dict
            The serialised variable.
        """
        pass

    # This may need to move to a different class
    @property
    @abstractmethod
    def grib_keys(self) -> dict[str, Any]:
        """Get the GRIB keys for the variable."""
        pass

    @abstractmethod
    def retrieval_metadata(self, repository: str) -> dict[str, Any] | None:
        """Return the request metadata stored for a data repository.

        This is the raw block collected at dataset-create time (e.g. the
        ``"mars"`` request); :meth:`retrieval_request` turns it into an
        actual request via the repository's
        :class:`~anemoi.transform.variables.retrieval.Retrieval`.

        Parameters
        ----------
        repository : str
            The name of the data repository / archival system (e.g. ``"mars"``).

        Returns
        -------
        dict or None
            The stored metadata, or None when the variable carries none
            for that repository.
        """
        pass

    def retrieval_request(self, repository: str = "mars") -> dict[str, Any] | None:
        """Build a retrieval request for a data repository / archival system.

        ``"mars"`` is one such repository; new ones are added by
        registering a
        :class:`~anemoi.transform.variables.retrieval.Retrieval` (see
        :func:`~anemoi.transform.variables.retrieval.register_retrieval`),
        so a dataset built from one repository can later be re-retrieved
        from another.

        Parameters
        ----------
        repository : str, optional
            The name of the data repository, by default ``"mars"``.

        Returns
        -------
        dict or None
            The retrieval request, or None when the variable carries no
            metadata for that repository.

        Raises
        ------
        ValueError
            If no retrieval system is registered under ``repository``.
        """
        from anemoi.transform.variables.retrieval import retrieval_system

        return retrieval_system(repository).request(self)

    @property
    @abstractmethod
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        pass

    @property
    @abstractmethod
    def units(self):
        """Get the units of the variable."""
        pass

    def compatible(
        self,
        other: Any,
        return_reason: bool = False,
        ignore_units: Any = False,
        ignore_time_processing: Any = False,
        ignore_processing_period: Any = False,
        ignore_type_of_level: Any = False,
    ) -> bool | tuple[bool, str | None]:
        """Check if two variables are compatible.

        Parameters
        ----------
        other : Any
            The other variable to compare with.
        return_reason : bool, optional
            If True, return a tuple of (bool, str) with the reason for incompatibility.
            Default is False.
        ignore_units : bool or str or list, optional
            Don't check units. Can be a boolean, a variable name, or a list of variable names.
            Default is False.
        ignore_time_processing : bool or str or list, optional
            Don't check time processing (e.g. whether the data is instantaneous or accumulated).
            Can be a boolean, a variable name, or a list of variable names.
            Default is False.
        ignore_processing_period : bool or str or list, optional
            Don't check time processing period (e.g. whether the data are 3-hourly or 6-hourly accumulations).
            Can be a boolean, a variable name, or a list of variable names.
            Default is False.
        ignore_type_of_level : bool or str or list, optional
            Don't check type of level (e.g. whether the data are on pressure levels or model levels).
            Can be a boolean, a variable name, or a list of variable names.
            Default is False.

        Returns
        -------
        bool or tuple[bool, str | None]
            If return_reason is False, returns True if compatible, False otherwise.
            If return_reason is True, returns a tuple of (bool, str | None) where the string
            is the reason for incompatibility, or None if compatible.
        """

        assert self.name == other.name
        name = self.name

        def _ignore(what, ignore):
            """Resolve an ignore option (bool, name or list of names) for this variable."""
            match ignore:
                case bool():
                    return ignore

                case str():
                    return name == ignore

                case list() | tuple() | set():
                    return name in ignore

                case _:
                    raise ValueError(
                        f"Invalid value for option '{what}': {ignore}. Expected a boolean, a string or a list of variable names."
                    )

        check_units = not _ignore("ignore_units", ignore_units)
        check_time_processing = not _ignore("ignore_time_processing", ignore_time_processing)
        check_period = not _ignore("ignore_processing_period", ignore_processing_period)
        check_type_of_level = not _ignore("ignore_type_of_level", ignore_type_of_level)

        def _compare():
            """Return the first incompatibility reason, or None when compatible."""
            if check_units:
                if self.units != other.units:
                    if self.units is None or other.units is None:
                        LOG.warning(
                            f"{self}: one of the variables has missing units: {self.units} vs {other.units}. Assuming they are compatible."
                        )
                    else:
                        return (
                            f"Units are not compatible: "
                            f"{self.units} (canonical: {self.units:c}) vs {other.units} (canonical: {other.units:c})"
                        )

            if check_time_processing:
                if self.time_processing != other.time_processing:
                    if self.time_processing is None or other.time_processing is None:
                        LOG.warning(
                            f"{self}: time processing types are not compatible: {self.time_processing} vs {other.time_processing}. Ignoring this incompatibility."
                        )
                    else:
                        return f"Time processinging types are not compatible: {self.time_processing} vs {other.time_processing}"

            if check_period:
                if self.period != other.period:
                    if self.period is None or other.period is None:
                        LOG.warning(
                            f"{self}: periods are not compatible: {self.period} vs {other.period}. Ignoring this incompatibility."
                        )
                    else:
                        return f"Periods are not compatible: {self.period} vs {other.period}"

            if check_type_of_level:
                if self.is_pressure_level != other.is_pressure_level:
                    if self.is_pressure_level is None or other.is_pressure_level is None:
                        LOG.warning(
                            f"{self}: pressure level status is not compatible: {self.is_pressure_level} vs {other.is_pressure_level}. Ignoring this incompatibility."
                        )
                    else:
                        return f"Pressure level status is not compatible: {self.is_pressure_level} vs {other.is_pressure_level}"

                if self.is_model_level != other.is_model_level:
                    if self.is_model_level is None or other.is_model_level is None:
                        LOG.warning(
                            f"{self}: model level status is not compatible: {self.is_model_level} vs {other.is_model_level}. Ignoring this incompatibility."
                        )
                    else:
                        return f"Model level status is not compatible: {self.is_model_level} vs {other.is_model_level}"

                if self.is_surface_level != other.is_surface_level:
                    if self.is_surface_level is None or other.is_surface_level is None:
                        LOG.warning(
                            f"{self}: surface level status is not compatible: {self.is_surface_level} vs {other.is_surface_level}. Ignoring this incompatibility."
                        )
                    else:
                        return f"Surface level status is not compatible: {self.is_surface_level} vs {other.is_surface_level}"

        reason = _compare()
        if reason:
            return (False, reason) if return_reason else False

        return (True, None) if return_reason else True

    @classmethod
    def check_compatibility(cls, variables1: dict, variables2: dict, *args, **kwargs) -> None:
        """Check that two ``{name: Variable}`` mappings are compatible.

        Raises ``ValueError`` when the variable names differ or any pair is
        incompatible (see :meth:`compatible`).

        Parameters
        ----------
        variables1 : dict
            The first ``{name: Variable}`` mapping.
        variables2 : dict
            The second ``{name: Variable}`` mapping.
        *args : Any
            Dictionaries of options merged into the ``ignore_*`` keyword
            arguments passed to :meth:`compatible`.
        **kwargs : Any
            Options passed to :meth:`compatible` (e.g. ``ignore_units``).
        """
        options = {}
        for arg in args:
            if isinstance(arg, dict):
                options.update(arg)
            else:
                raise ValueError(f"Invalid argument: {arg}. Expected a dictionary.")

        options.update(kwargs)

        keys1 = set(variables1.keys())
        keys2 = set(variables2.keys())

        if keys1 != keys2:
            raise ValueError(f"Variable compatibility: missing={keys1-keys2}, added={keys2-keys1}")

        reasons = []
        for k in keys1:
            compatible, reason = variables1[k].compatible(variables2[k], return_reason=True, **options)
            if not compatible:
                reasons.append(f"{k}: {reason}")

        if reasons:
            raise ValueError(f"Variables are not compatible: {'; '.join(reasons)}")
