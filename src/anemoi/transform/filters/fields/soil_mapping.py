# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.data.field.component.vertical import Vertical

from anemoi.transform import Field
from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry

# Mapping of named soil parameters to their generic (param, levelist) representation.
#
# The named parameters (e.g. ``stl1``) encode the soil level in the parameter
# name itself and carry no ``levtype``/``levelist``. The generic representation
# (e.g. ``sot`` on ``levtype=sol``, ``levelist=1``) is the one used by MARS-like
# retrievals. This filter converts between the two.
#
#   named param -> {"param": generic param, "levelist": soil level}
SOIL_MAPPING: dict[str, dict[str, Any]] = {
    "stl1": {"param": "sot", "levelist": 1},
    "stl2": {"param": "sot", "levelist": 2},
    "stl3": {"param": "sot", "levelist": 3},
    "swvl1": {"param": "vsw", "levelist": 1},
    "swvl2": {"param": "vsw", "levelist": 2},
    "swvl3": {"param": "vsw", "levelist": 3},
}

# Soil ``levtype`` used by the generic representation in MARS requests.
SOIL_LEVTYPE = "sol"

# Soil level type used by the generic representation in the field's vertical
# component (the earthkit-data name for GRIB ``soilLayer``, i.e. ``levtype=sol``).
SOIL_LEVEL_TYPE = "soil_layer"


def _build_inverse_mapping(
    mapping: dict[str, dict[str, Any]],
) -> dict[tuple[str, int], str]:
    """Build the inverse mapping from ``(param, levelist)`` to a named soil parameter.

    Parameters
    ----------
    mapping : dict[str, dict[str, Any]]
        The forward mapping from named soil parameters to their generic
        ``(param, levelist)`` representation.

    Returns
    -------
    dict[tuple[str, int], str]
        A mapping from ``(param, levelist)`` tuples to the named soil parameter.
    """
    inverse: dict[tuple[str, int], str] = {}
    for named, spec in mapping.items():
        inverse[(spec["param"], int(spec["levelist"]))] = named
    return inverse


class SoilMapping(SingleFieldFilter):
    """A filter to map named soil parameters to their generic representation, and back.

    The forward transform maps named soil parameters (``stl1``, ``stl2``,
    ``stl3``, ``swvl1``, ``swvl2``, ``swvl3``) to their generic grib2
    representation, i.e. a generic ``param`` (``sot`` for soil temperature,
    ``vsw`` for volumetric soil water) on ``levtype=sol`` with the soil level
    encoded in ``levelist``.

    The backward transform reverses this, converting generic soil fields (with
    ``levtype``/``levelist``) back into the named soil parameters (grib1).

    Examples
    --------

    .. code-block:: yaml

        input:
          pipe:
            - source:
                ...
            - soil_to_levtype_sol:
                # no configuration needed

    """

    def prepare_filter(self) -> None:
        self.mapping = {K: {**v, "levtype": SOIL_LEVTYPE} for K, v in SOIL_MAPPING.copy().items()}
        self.inverse_mapping = _build_inverse_mapping(self.mapping)
        self.named_params = list(self.mapping)
        self.generic_params = sorted({spec["param"] for spec in self.mapping.values()})

    def forward_select(self) -> dict[str, list[str]]:
        # Select only the named soil parameters (which carry no levtype/levelist).
        return {"parameter.variable": self.named_params}

    def backward_select(self) -> dict[str, list[str]]:
        # Select only the generic soil parameters (which carry levtype/levelist).
        return {"parameter.variable": self.generic_params}

    def forward_transform(self, field: Field) -> Field:
        """Convert a named soil parameter to its generic representation.

        Parameters
        ----------
        field : Field
            The field with a named soil parameter (e.g. ``stl1``).

        Returns
        -------
        Field
            The field with the generic ``param``, ``levtype`` and ``levelist``.
        """
        spec = self.mapping[field.parameter.variable()]
        result = Field.from_numpy(field.to_numpy(), template=field, parameter={"variable": spec["param"]})
        # Named soil fields carry no vertical component, so it is replaced as a whole.
        return result.set(vertical=Vertical.from_dict({"level": spec["levelist"], "level_type": SOIL_LEVEL_TYPE}))

    def backward_transform(self, field: Field) -> Field:
        """Convert a generic soil parameter back to its named representation.

        Parameters
        ----------
        field : Field
            The field with a generic soil ``param`` and a ``levelist``.

        Returns
        -------
        Field
            The field with the named soil parameter metadata (e.g. ``stl1``).
        """
        param = field.parameter.variable()
        levelist = field.get("vertical.level", default=None)

        if levelist is None:
            # Not enough information to map back, pass through unchanged.
            return field

        named = self.inverse_mapping.get((param, int(levelist)))
        if named is None:
            return field
        result = Field.from_numpy(field.to_numpy(), template=field, parameter={"variable": named})
        return result.set(vertical={"level": None, "level_type": None})

    def patch_data_request(self, data_request: dict[str, Any]) -> dict[str, Any]:
        """Modify the data request to retrieve the generic soil parameters.

        Named soil parameters (``stl1`` ...) are replaced with their generic
        parameter, the corresponding soil ``levelist`` is added and the
        ``levtype`` is set to ``sol``.

        Parameters
        ----------
        data_request : dict[str, Any]
            The original data request.

        Returns
        -------
        dict[str, Any]
            The modified data request.
        """
        param = data_request.get("param")
        if param is None:
            return data_request

        params = param if isinstance(param, list) else [param]

        named = [p for p in params if p in self.mapping]
        if not named:
            return data_request

        remaining = [p for p in params if p not in self.mapping]

        new_params = list(remaining)
        levelists: list[int] = []
        for p in named:
            spec = self.mapping[p]
            if spec["param"] not in new_params:
                new_params.append(spec["param"])
            if spec["levelist"] not in levelists:
                levelists.append(spec["levelist"])

        data_request["param"] = new_params
        data_request["levtype"] = SOIL_LEVTYPE

        existing_levelist = data_request.get("levelist")
        if existing_levelist:
            existing = existing_levelist if isinstance(existing_levelist, list) else [existing_levelist]
            for lvl in existing:
                if lvl not in levelists:
                    levelists.append(lvl)
        data_request["levelist"] = sorted(levelists)

        return data_request


filter_registry.register("soil_to_levtype_sol", SoilMapping)
filter_registry.register("levtype_sol_to_soil", SoilMapping.reversed)
