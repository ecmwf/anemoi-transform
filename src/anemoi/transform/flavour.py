# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from typing import Mapping

from anemoi.utils.rules import Rule
from anemoi.utils.rules import RuleSet

from anemoi.transform.fields import MISSING_METADATA
from anemoi.transform.fields import Flavour
from anemoi.transform.fields import new_flavoured_field


class RuleBasedFlavour(Flavour):
    """Rule-based flavour for GRIB files."""

    def __init__(self, rules):
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

        print(self.rules)

    def apply(self, field):
        return new_flavoured_field(field, self)

    def __call__(self, key, field):
        """Called when the field metadata is queried"""

        if key not in self.rules:
            return MISSING_METADATA

        class FieldMetadata(Mapping): ...

        for rule in self.rules[key]:
            if rule.match(field.metadata()):
                return rule.result

        return MISSING_METADATA
