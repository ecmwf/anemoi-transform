# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple

LOG = logging.getLogger(__name__)


def _lost(f: Any) -> None:
    """Raise a ValueError indicating a lost field.

    Parameters
    ----------
    f : Any
        The lost field.
    """
    raise ValueError(f"Lost field {f}")


def _flatten(params: List[Any]) -> List[str]:
    """Flatten a list of parameters.

    Parameters
    ----------
    params : list of Any
        List of parameters to flatten.

    Returns
    -------
    list of str
        Flattened list of parameters.
    """
    flat = []
    for p in params:
        if isinstance(p, (list, tuple)):
            flat.extend(_flatten(p))
        else:
            flat.append(p)
    return flat


class GroupByParam:
    """Group matching fields by parameters name.

    Parameters
    ----------
    params : list of str
        List of parameters to group by.
    """

    def __init__(self, params: List[str]) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = _flatten(params)

    def _get_groups(self, data: List[Any], *, other: Callable[[Any], None] = _lost) -> None:
        assert callable(other), type(other)
        self.groups: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = defaultdict(dict)
        self.groups_params = set()
        for f in data:
            key = f.metadata(namespace="mars")
            if not key:
                keys = [k for k in f.metadata().keys() if k not in ("latitudes", "longitudes", "values")]
                key = {k: f.metadata(k) for k in keys}
                if not keys:
                    raise NotImplementedError(f"GroupByParam: {f} has no sufficient metadata")

            param = key.pop("param", f.metadata("param"))

            if param not in self.params:
                other(f)
                continue

            key = tuple(key.items())

            if param in self.groups[key]:
                raise ValueError(f"Duplicate component {param} for {key}")
            self.groups[key][param] = f
            self.groups_params.add(param)
        LOG.info(f"Params groups: {self.groups_params}")

    def iterate(self, data: List[Any], *, other: Callable[[Any], None] = _lost) -> Iterator[Tuple[Any, ...]]:
        """Iterate over the data and group fields by parameters.

        Parameters
        ----------
        data : list of Any
            List of data fields to group.
        other : callable, optional
            Function to call for fields that do not match the parameters, by default _lost.

        Returns
        -------
        Iterator[Tuple[Any, ...]]
            Iterator yielding tuples of grouped fields.
        """
        self._get_groups(data, other=other)
        for _, group in self.groups.items():
            if len(group) != len(self.params):
                for p in data:
                    print(p)
                raise ValueError(f"Missing component. Want {sorted(self.params)}, got {sorted(self.groups.keys())}")

            yield tuple(group[p] for p in self.params)
