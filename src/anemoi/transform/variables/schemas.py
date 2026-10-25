# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Pydantic models of the serialised variable layouts.

Each supported ``"schema"`` value has a model here; deserialisation
(:meth:`Variable.from_dict`) validates the incoming dictionary against
the model of the class it dispatches to, and serialisation
(:meth:`Variable.as_dict`) is a ``model_dump`` — so both directions of
the JSON round-trip are validated.

All models allow extra keys: a newer writer of the *same* schema may add
keys without breaking older readers (incompatible layout changes must
use a new ``"schema"`` value instead).
"""

import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Annotated
from typing import Any
from typing import Literal

from anemoi.utils.dates import as_timedelta
from anemoi.utils.dates import frequency_to_string
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field as PydanticField
from pydantic import PlainSerializer

LOG = logging.getLogger(__name__)

# A timedelta that accepts every spelling found in existing metadata
# ("6h", "6:00:00", integer hours) and serialises to the short form.
Timedelta = Annotated[
    timedelta,
    BeforeValidator(as_timedelta),
    PlainSerializer(frequency_to_string, return_type=str),
]


def _tolerant_period(value: Any) -> Any:
    """Treat a malformed legacy ``period`` (not a 2-element sequence) as missing.

    Parameters
    ----------
    value : Any
        The raw ``period`` value from the serialised variable.

    Returns
    -------
    Any
        The value when it is a 2-element sequence, None otherwise.
    """
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        LOG.warning("Ignoring malformed variable period: %r", value)
        return None
    return value


Period = Annotated[tuple[Timedelta, Timedelta] | None, BeforeValidator(_tolerant_period)]


class VariableSchema(BaseModel):
    """Keys shared by every serialised variable layout.

    The time-processing keys and the create-time flags live at the top
    level in every schema — usage-time shims (e.g. the constant-fields
    overlay in anemoi-datasets) mutate them on the raw dictionaries
    without knowing the schema.
    """

    model_config = ConfigDict(extra="allow")

    process: str | None = None
    period: Period = None
    computed_forcing: bool = False
    constant_in_time: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialise back to a JSON-compatible dictionary.

        Only explicitly-set keys are emitted (``exclude_unset``), so a
        validated dictionary round-trips without gaining default values.

        Returns
        -------
        dict
            The serialised variable.
        """
        return self.model_dump(mode="json", by_alias=True, exclude_unset=True)


class LegacyVariableSchema(VariableSchema):
    """The historical MARS-vocabulary layout (dictionaries without a ``"schema"`` key).

    Written by every anemoi-datasets version before the ``variable/1``
    schema; present in all existing datasets and checkpoints.
    """

    mars: dict[str, Any] = PydanticField(default_factory=dict)
    units: str | None = None
    grib: dict[str, Any] | None = None


class ParameterComponent(BaseModel):
    """The ``parameter`` component of the ``variable/1`` schema."""

    model_config = ConfigDict(extra="allow")

    variable: str
    units: str | None = None


class VerticalComponent(BaseModel):
    """The ``vertical`` component of the ``variable/1`` schema."""

    model_config = ConfigDict(extra="allow")

    level_type: str | None = None
    level: int | float | str | None = None


class VariableSchemaV1(VariableSchema):
    """The ``variable/1`` layout: attributes under earthkit-data component names.

    Data from any earthkit-data source (GRIB, NetCDF, ...) serialises
    the same way. The MARS request, when the source provides one, is
    kept under ``mars`` (a sanctioned legacy surface — it feeds MARS
    request generation and GRIB encoding).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_: Literal["variable/1"] = PydanticField(alias="schema")
    parameter: ParameterComponent
    vertical: VerticalComponent | None = None
    grib: dict[str, Any] | None = None
    mars: dict[str, Any] | None = None
