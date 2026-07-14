# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Variables described by a live :class:`~anemoi.transform.fields.Field`.

This is the collection side of the variable pipeline: at dataset-create
time a representative field of each variable is turned into a
:class:`VariableFromField`, and :meth:`VariableFromField.as_dict`
produces the ``variable/1`` serialisation (see
:mod:`anemoi.transform.variables.components`) that is stored in the
dataset metadata and later deserialised by :meth:`Variable.from_dict`.
"""

import logging
import warnings
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from anemoi.utils.dates import as_timedelta

from anemoi.transform.fields import metadata_key
from anemoi.transform.variables.components import SCHEMA
from anemoi.transform.variables.components import ComponentVariable
from anemoi.transform.variables.schemas import VariableSchemaV1

if TYPE_CHECKING:
    from datetime import timedelta

LOG = logging.getLogger(__name__)

# The MARS-style GRIB keys reported by ``grib_keys``, with their component
# paths. GRIB encoding is a sanctioned legacy surface (the MARS keys are the
# output format there), so the deprecation warning is silenced while deriving.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    _GRIB_KEYS = {key: metadata_key(key) for key in ("param", "levtype", "levelist", "step", "number")}

# https://codes.ecmwf.int/grib/format/grib2/ctables/4/10/
_TYPE_OF_STATISTICAL_PROCESSING: dict[int | None, str | None] = {
    None: None,
    0: "average",
    1: "accumulation",
    2: "maximum",
    3: "minimum",
    4: "difference(end-start)",
    5: "root_mean_square",
    6: "standard_deviation",
    7: "covariance",
    8: "difference(start-end)",
    9: "ratio",
    10: "standardized_anomaly",
    11: "summation",
    100: "severity",
    101: "mode",
}

# https://codes.ecmwf.int/grib/format/grib1/ctable/5/
_TIME_RANGE_INDICATOR: dict[int, str] = {
    4: "accumulation",
    3: "average",
}

_STEP_TYPE_FOR_CONVERSION: dict[str, str] = {
    "min": "minimum",
    "max": "maximum",
    "accum": "accumulation",
    "avg": "average",
}

# Parameters whose GRIB encoding does not identify the statistical
# process (not in the param db).
_PROCESS_PATCHES: dict[str, str] = {
    "10fg6": "maximum",
    "mntpr3": "minimum",
    "mntpr6": "minimum",
    "mxtpr3": "maximum",
    "mxtpr6": "maximum",
}


class VariableFromField(ComponentVariable):
    """A variable described by a :class:`~anemoi.transform.fields.Field`.

    The variable's attributes are read from the field's components on
    demand — there is no intermediate metadata dictionary or delegate
    object.
    """

    def __init__(self, name: str, field: Any) -> None:
        """Initialize the variable with a name and a field.

        Parameters
        ----------
        name : str
            The name of the variable.
        field : Field
            The field describing the variable.
        """
        super().__init__(name)
        self.field = field

    def _component(self, path: str) -> Any:
        """Resolve a dotted earthkit-data component path on the field.

        Parameters
        ----------
        path : str
            The component path (e.g. ``"parameter.units"``).

        Returns
        -------
        Any
            The value, or None when the component is missing.
        """
        return self.field.get(path, default=None)

    @cached_property
    def _mars(self) -> dict[str, Any]:
        """The field's MARS request metadata (empty when the source has none).

        Falls back to the ``metadata.default`` collection when the MARS
        one is empty, and repairs unusable ``param`` values (``"~"``,
        ``"unknown"``) from the raw GRIB keys.
        """
        md = self.field.get(collections="metadata.mars")
        if not md:
            md = self.field.get(collections="metadata.default")
        if md is None:
            md = {}

        md = {k: v for k, v in md.items() if not k.startswith("_")}

        if md.get("param") == "~":
            md["param"] = self.field.metadata("param")
            assert md["param"] not in ("~", "unknown"), (md, self.field.metadata("param"))

        if md.get("param") == "unknown":
            md["param"] = str(self.field.get("metadata.paramId", default="unknown"))

        return md

    @cached_property
    def _time_window(self) -> tuple[str | None, Union["timedelta", None], Union["timedelta", None]]:
        """The variable's time processing as ``(process, start, end)``.

        The window is recovered from the raw GRIB step keys
        (``startStep``/``endStep``), with fixes for mis-encoded GRIB1
        accumulations, or — for in-memory fields that carry no raw GRIB
        keys — from the ``proc.*`` time-processing components. When
        there is no window, the process falls back to the
        ``time.statistical_process`` component (usually None).

        Raises
        ------
        ValueError
            If the field has a time window whose statistical process
            cannot be established.
        """
        field = self.field

        startStep = field.get("metadata.startStep", default=None)
        if startStep is not None:
            startStep = as_timedelta(startStep)

        endStep = field.get("metadata.endStep", default=None)
        if endStep is not None:
            endStep = as_timedelta(endStep)

        stepTypeForConversion = field.get("metadata.stepTypeForConversion", default=None)
        typeOfStatisticalProcessing = field.get("metadata.typeOfStatisticalProcessing", default=None)
        timeRangeIndicator = field.get("metadata.timeRangeIndicator", default=None)

        # GRIB1 precipitation accumulations are not correctly encoded
        if startStep == endStep and stepTypeForConversion == "accum":
            # in such case of incorrect encoding, P1 refers to endStep and P2 to startStep.
            # Note that this is, on purpose, the opposite of the usual convention.
            endStep = as_timedelta(field.metadata("P1"))
            startStep = as_timedelta(field.metadata("P2"))

        if startStep is None and endStep is None:
            # In-memory fields (e.g. from the accumulate source) do not carry
            # raw GRIB keys; recover the window from the time-processing and
            # time components instead.
            method = field.get("proc.time_method", default=None)
            span = field.get("proc.time_value", default=None)
            step = field.get("time.step", default=None)
            if method is not None and method != "instant" and span is not None and step is not None:
                endStep = as_timedelta(step)
                startStep = endStep - as_timedelta(span)
                stepTypeForConversion = str(method)

        if startStep is not None and endStep is not None:
            assert endStep >= startStep, (startStep, endStep, self._mars)

        if startStep == endStep:
            return self._component("time.statistical_process"), None, None

        process = _TYPE_OF_STATISTICAL_PROCESSING.get(typeOfStatisticalProcessing)
        if process is None:
            process = _TIME_RANGE_INDICATOR.get(timeRangeIndicator)
        if process is None:
            process = _STEP_TYPE_FOR_CONVERSION.get(stepTypeForConversion)
        if process is None:
            param = self._mars.get("param", self.param)
            process = _PROCESS_PATCHES.get(param)
            if process is not None:
                LOG.error(f"Unknown process {stepTypeForConversion} for {param}, using {process} instead")

        if process is None:
            raise ValueError(
                f"Unknown for {self._mars.get('param', self.param)}:"
                f" {stepTypeForConversion=} ({_STEP_TYPE_FOR_CONVERSION.get(stepTypeForConversion)}),"
                f" {typeOfStatisticalProcessing=} ({_TYPE_OF_STATISTICAL_PROCESSING.get(typeOfStatisticalProcessing)}),"
                f" {timeRangeIndicator=} ({_TIME_RANGE_INDICATOR.get(timeRangeIndicator)})"
            )

        return process, startStep, endStep

    @property
    def is_constant_in_time(self) -> bool | None:
        """Check if the variable is constant in time (unknown for a single field)."""
        return None

    @property
    def is_computed_forcing(self) -> bool | None:
        """Check if the variable is a computed forcing (unknown for a single field)."""
        return None

    @property
    def time_processing(self):
        """Get the time processing type of the variable (e.g. ``"accumulation"``)."""
        return self._time_window[0]

    @cached_property
    def _is_point_in_time(self) -> bool:
        """Whether the field positively declares a zero-length validity window.

        True for a zero-length raw GRIB step range or an explicit
        ``instant`` proc method. A field carrying neither says nothing
        about its window (e.g. in-memory or wrapped fields), so False
        here means "unknown", not "valid over a period".
        """
        start = self.field.get("metadata.startStep", default=None)
        end = self.field.get("metadata.endStep", default=None)
        if start is not None and start == end:
            return True
        return self.field.get("proc.time_method", default=None) == "instant"

    @property
    def period(self) -> Union["timedelta", None]:
        """Get the variable's period as a timedelta.

        Returns a timedelta of 0 only when the field positively declares
        itself instantaneous; ``None`` when the window is unknown (a
        field without time-processing metadata cannot be distinguished
        from an instantaneous one, and a wrong 0 would fail the
        compatibility check against metadata that knows the period).
        """
        process, start, end = self._time_window
        if process is not None:
            if start is None or end is None:
                return None
            return end - start

        if self._is_point_in_time:
            return as_timedelta(0)

        return None

    # This may need to move to a different class
    @property
    def grib_keys(self) -> dict[str, Any]:
        """Get MARS-style GRIB keys for the variable, from the field's components."""
        from anemoi.transform.variables.components import LEVEL_TYPE_MAPPING

        keys = {}
        for grib_key, path in _GRIB_KEYS.items():
            value = self.field.get(path, default=None)
            if value is not None:
                keys[grib_key] = value
        if "levtype" in keys:
            keys["levtype"] = LEVEL_TYPE_MAPPING.get(keys["levtype"], keys["levtype"])
        return keys

    def as_dict(self) -> dict[str, Any]:
        """Serialise the variable to the ``variable/1`` schema.

        The result is validated against
        :class:`~anemoi.transform.variables.schemas.VariableSchemaV1`
        before being dumped, is JSON-compatible, and deserialises to a
        :class:`~anemoi.transform.variables.components.VariableFromComponents`
        via :meth:`Variable.from_dict`. Sections with nothing to record
        (``vertical``, ``process``/``period``, ``grib``, ``mars``) are
        omitted.

        Returns
        -------
        dict
            The serialised variable.
        """
        from anemoi.transform.units import BUILD_UNIT_ALIASES

        result: dict[str, Any] = {"schema": SCHEMA}

        parameter: dict[str, Any] = {"variable": self.param}
        # Prefer the raw GRIB units string (as the legacy collection did) and
        # normalise the WMO fraction/percent spellings; wrapped fields mask
        # metadata.units, so fall back to the richer parameter.units component.
        units = self.field.get("metadata.units", default=None)
        if units is None:
            pint_units = self._component("parameter.units")
            if pint_units is not None:
                units = str(pint_units)
        units = BUILD_UNIT_ALIASES.get(units, units)
        if units is not None:
            parameter["units"] = units
        result["parameter"] = parameter

        vertical: dict[str, Any] = {}
        level_type = self._component("vertical.level_type")
        if level_type is not None:
            vertical["level_type"] = str(level_type)
        if (level := self.level) is not None:
            vertical["level"] = level
        if vertical:
            result["vertical"] = vertical

        process, start, end = self._time_window
        if process is not None:
            result["process"] = str(process)
        if start is not None and end is not None:
            result["period"] = (start, end)

        grib = {}
        for key in ("paramId", "shortName"):
            value = self.field.get(f"metadata.{key}", default=None)
            if value is not None:
                grib[key] = value
        if grib:
            result["grib"] = grib

        if self._mars:
            result["mars"] = dict(self._mars)

        return VariableSchemaV1.model_validate(result).as_dict()
