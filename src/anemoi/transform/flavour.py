# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from typing import Any

import earthkit.data as ekd
from anemoi.utils.rules import Rule
from anemoi.utils.rules import RuleSet

from anemoi.transform.fields import MISSING_METADATA
from anemoi.transform.fields import Flavour
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.fields import new_flavoured_field
from anemoi.transform.metadata import get_metadata


class _FieldMetadataMapping:
    """A Mapping-like wrapper around a field, for use with Rule.match().

    Rule.match() expects a Mapping, using ``key in obj`` and ``obj[key]``.
    earthkit-data fields support neither, and earthkit-data 1.0 removed the
    no-argument ``field.metadata()`` call that used to return a dict-like object,
    so the field has to be adapted.

    Keys are resolved with :func:`anemoi.transform.metadata.get_metadata`, so
    flavour rules can be written against raw metadata keys (e.g. GRIB keys such as
    ``paramId``) as well as MARS-style keys (e.g. ``param``, ``levtype``).
    """

    _MISSING = object()

    def __init__(self, field: ekd.Field) -> None:
        self._field = field

    def __contains__(self, key: str) -> bool:
        return get_metadata(self._field, key, default=self._MISSING) is not self._MISSING

    def __getitem__(self, key: str) -> Any:
        return get_metadata(self._field, key)


class RuleBasedFlavour(Flavour):
    """Rule-based flavour for GRIB files."""

    def __init__(self, rules: RuleSet | list[Rule] | dict[str, Any]):
        """Initialize the RuleBasedFlavour with a set of rules.

        Parameters
        ----------
        rules : Union[RuleSet, List[Rule], Dict[str, Any]]
            The rules to be applied, which can be a RuleSet, a list of Rule objects,
            or a dictionary mapping keys to rule definitions.
        """
        rules = RuleSet.from_any(rules)
        per_target = defaultdict(list)
        for rule in rules:
            result = rule.result
            assert isinstance(result, dict), "Expected a dictionary as result."
            for key, value in result.items():
                per_target[key].append(Rule(rule.condition, value))

        self.rules = {}
        for key, value in per_target.items():
            self.rules[key] = RuleSet.from_any(value)

    def apply(self, field: ekd.Field) -> ekd.Field:
        """Apply the flavour to a single field.

        Parameters
        ----------
        field : ekd.Field
            The field to which the flavour will be applied.

        Returns
        -------
        ekd.Field
            The field with the applied flavour.
        """
        return new_flavoured_field(field, self)

    def map(self, fieldlist: ekd.FieldList) -> ekd.FieldList:
        """Apply the flavour to a fieldlist.

        Parameters
        ----------
        fieldlist : ekd.FieldList
            The list of fields to which the flavour will be applied.

        Returns
        -------
        ekd.FieldList
            The list of fields with the applied flavour.
        """
        return new_fieldlist_from_list([self.apply(field) for field in fieldlist])

    def keys(self) -> Any:
        """The metadata keys this flavour may override.

        Returns
        -------
        Any
            The keys targeted by the flavour rules.
        """
        return self.rules.keys()

    def __call__(self, key: str, field: ekd.Field) -> Any:
        """Called when the field metadata is queried.

        Parameters
        ----------
        key : str
            The metadata key being queried.
        field : ekd.Field
            The field whose metadata is being queried.

        Returns
        -------
        Any
            The result of the metadata query, or MISSING_METADATA if no match is found.
        """
        if key not in self.rules:
            return MISSING_METADATA

        for rule in self.rules[key]:
            if rule.match(_FieldMetadataMapping(field)):
                return rule.result

        return MISSING_METADATA
