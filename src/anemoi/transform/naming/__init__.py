# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Field naming schemes.

A naming scheme computes a unique name for each field (e.g. ``"t_850"``)
and attaches it to the field as the ``labels.name`` label, so that the
name can be used directly when filtering and sorting field collections
(``sel``, ``order_by``, ``to_cube`` on the ``labels.name`` key).

The naming configuration is chosen when a training dataset is built
(anemoi-datasets), serialised in the dataset metadata (and hence in
checkpoints), and deserialised at inference time (anemoi-inference) so
that both sides name fields identically.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any

from anemoi.utils.registry import Registry

from anemoi.transform import Field
from anemoi.transform import FieldList

naming_registry = Registry(__name__)


class Naming(ABC):
    """Base class for all field naming schemes."""

    @abstractmethod
    def name(self, field: Field) -> str:
        """Return the name of a field.

        Parameters
        ----------
        field : Field
            The field to name.

        Returns
        -------
        str
            The name of the field.
        """
        pass

    def __call__(self, fields: FieldList) -> FieldList:
        """Attach each field's name as its ``labels.name`` label.

        Fields that already carry a name (e.g. explicitly renamed with the
        ``rename`` filter, which uses :meth:`Field.with_name`) keep it.

        Parameters
        ----------
        fields : FieldList
            The fields to name.

        Returns
        -------
        FieldList
            A new field list where every field carries its name in
            ``labels.name``.
        """
        result = []
        for field in fields:
            if field.get("labels.name", default=None) is None:
                field = Field.with_name(field, self.name(field))
            result.append(field)
        return FieldList.from_fields(result)


def create_naming(config: Any) -> Naming:
    """Create a naming scheme from the given configuration.

    Parameters
    ----------
    config : Any
        The configuration for the naming scheme: the name of a registered
        scheme (e.g. ``"param"``, ``"param_levelist"``), a template string
        containing ``{key}`` variables (e.g. ``"{param}_{levelist}"``), or a
        registry config dictionary.

    Returns
    -------
    Naming
        The created naming scheme.
    """
    from anemoi.transform.naming.template import TemplateNaming

    if isinstance(config, str):
        if config == "default":
            config = "param_levelist"
        if "{" in config:
            return TemplateNaming(config)
    return naming_registry.from_config(config)


def variable_naming_from_remapping(remapping: Any) -> str | None:
    """Return the ``variable_naming`` equivalent of a legacy ``remapping``.

    Dataset recipes and checkpoint metadata used to carry the field naming
    as an earthkit-data remapping dictionary, e.g.
    ``{"param_level": "{param}_{levelist}"}``. This extracts the
    ``param_level`` entry (the only one that ever named fields) as a
    ``variable_naming`` configuration.

    Parameters
    ----------
    remapping : Any
        The legacy remapping configuration (a dictionary, possibly empty).

    Returns
    -------
    str or None
        The equivalent ``variable_naming`` configuration, or ``None`` when
        the remapping carries no naming information.
    """
    if not remapping:
        return None
    return remapping.get("param_level")


def create_naming_from_remapping(remapping: Any) -> Naming | None:
    """Create a naming scheme from a legacy ``remapping`` configuration.

    See :func:`variable_naming_from_remapping`.

    Parameters
    ----------
    remapping : Any
        The legacy remapping configuration (a dictionary, possibly empty).

    Returns
    -------
    Naming or None
        The equivalent naming scheme, or ``None`` when the remapping
        carries no naming information.
    """
    config = variable_naming_from_remapping(remapping)
    if config is None:
        return None
    return create_naming(config)
