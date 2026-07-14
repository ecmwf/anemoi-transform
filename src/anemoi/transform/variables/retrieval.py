# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Data repositories that variables can be retrieved from.

A *retrieval system* is a data archive, database or repository (MARS,
and others to come) from which the gridded data behind a variable can be
fetched. Each system is registered under a name in ``RETRIEVAL_SYSTEMS``
so new data-access methods can be added — from this or a downstream
package — without touching the :class:`~anemoi.transform.variables.Variable`
classes. A system knows how to:

* **collect** its request metadata from a live
  :class:`~anemoi.transform.fields.Field` at dataset-create time
  (:meth:`Retrieval.collect`), so it is serialised into the variable's
  metadata alongside the earthkit-data components;
* **build a request** from a (deserialised) variable
  (:meth:`Retrieval.request`), so a dataset built from one repository
  carries enough metadata to be re-retrieved later — possibly from a
  different repository.

The MARS block that anemoi-inference reads to generate retrieval
requests (see ``Metadata.simple_mars_requests``) is the ``"mars"``
system implemented here.
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

if TYPE_CHECKING:
    from anemoi.transform.fields import Field
    from anemoi.transform.variables import Variable

LOG = logging.getLogger(__name__)

# Retrieval system name → singleton instance. Populated at import time by
# ``register_retrieval`` (no lazy mutation, so no lock is required).
RETRIEVAL_SYSTEMS: dict[str, "Retrieval"] = {}


def register_retrieval(retrieval: "Retrieval") -> "Retrieval":
    """Register a retrieval system under its ``name``.

    Parameters
    ----------
    retrieval : Retrieval
        The retrieval system instance to register.

    Returns
    -------
    Retrieval
        The registered instance (so the call can be used as an expression).
    """
    RETRIEVAL_SYSTEMS[retrieval.name] = retrieval
    return retrieval


def retrieval_system(name: str) -> "Retrieval":
    """Look up a registered retrieval system by name.

    Parameters
    ----------
    name : str
        The name of the retrieval system (e.g. ``"mars"``).

    Returns
    -------
    Retrieval
        The registered retrieval system.

    Raises
    ------
    ValueError
        If no retrieval system is registered under ``name``.
    """
    try:
        return RETRIEVAL_SYSTEMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown retrieval system {name!r} (known: {retrieval_systems()})."
            " Register it with 'register_retrieval'."
        )


def retrieval_systems() -> list[str]:
    """List the names of the registered retrieval systems.

    Returns
    -------
    list of str
        The sorted retrieval system names.
    """
    return sorted(RETRIEVAL_SYSTEMS)


class Retrieval(ABC):
    """A data repository / archival system that variables can be retrieved from."""

    name: ClassVar[str]

    def collect(self, field: "Field") -> dict[str, Any] | None:
        """Extract this system's request metadata from a live field.

        Called at dataset-create time; the returned block is serialised
        into the variable's metadata under :attr:`name`. The default
        collects nothing (a system that cannot be recovered from a field).

        Parameters
        ----------
        field : Field
            The field describing the variable.

        Returns
        -------
        dict or None
            The request metadata, or None when the field carries none.
        """
        return None

    @abstractmethod
    def request(self, variable: "Variable") -> dict[str, Any] | None:
        """Build a retrieval request for a variable.

        Parameters
        ----------
        variable : Variable
            The variable to build a request for.

        Returns
        -------
        dict or None
            The retrieval request, or None when the variable carries no
            metadata for this system.
        """
        pass


class MarsRetrieval(Retrieval):
    """The MARS archive — the historical retrieval system.

    Its metadata block (``"mars"``) is the MARS request that
    anemoi-inference expands into per-date retrieval requests and that
    feeds GRIB encoding.
    """

    name = "mars"

    def collect(self, field: "Field") -> dict[str, Any] | None:
        """Collect the field's MARS request metadata.

        Falls back to the ``metadata.default`` collection when the MARS
        one is empty, drops private (underscore-prefixed) keys and
        repairs unusable ``param`` values (``"~"``, ``"unknown"``) from
        the raw GRIB keys.

        Parameters
        ----------
        field : Field
            The field describing the variable.

        Returns
        -------
        dict or None
            The MARS request metadata, or None when the field has none.
        """
        md = field.get(collections="metadata.mars")
        if not md:
            md = field.get(collections="metadata.default")
        if not md:
            return None

        md = {k: v for k, v in md.items() if not k.startswith("_")}

        if md.get("param") == "~":
            md["param"] = field.metadata("param")
            assert md["param"] not in ("~", "unknown"), (md, field.metadata("param"))

        if md.get("param") == "unknown":
            md["param"] = str(field.get("metadata.paramId", default="unknown"))

        return md or None

    def request(self, variable: "Variable") -> dict[str, Any] | None:
        """Build a MARS request from the variable's stored ``mars`` block.

        Parameters
        ----------
        variable : Variable
            The variable to build a request for.

        Returns
        -------
        dict or None
            A copy of the stored MARS request, or None when absent.
        """
        block = variable.retrieval_metadata(self.name)
        return dict(block) if block else None


register_retrieval(MarsRetrieval())
