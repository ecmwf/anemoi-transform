# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from typing import Any
from typing import Callable
from typing import List


def _lost(f):
    raise ValueError(f"Lost field {f}")


class GroupByMarsParam:
    """Group matching fields by MARS paramters name."""

    def __init__(self, params: List[str]) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = params

    def iterate(self, data: List[Any], *, other: Callable[[Any], None] = _lost) -> Any:

        assert callable(other), type(other)
        groups = defaultdict(dict)

        for f in data:
            key = f.metadata(namespace="mars")
            param = key.pop("param")

            if param not in self.params:
                other(f)
                continue

            key = tuple(key.items())

            if param in groups[key]:
                raise ValueError(f"Duplicate component {param} for {key}")

            groups[key][param] = f

        for _, group in groups.items():
            if len(group) != len(self.params):
                for p in data:
                    print(p)
                raise ValueError(f"Missing component. Want {sorted(self.params)}, got {sorted(group.keys())}")

            yield tuple(group[p] for p in self.params)
