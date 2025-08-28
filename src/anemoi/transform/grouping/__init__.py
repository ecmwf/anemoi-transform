# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

from earthkit.data import SimpleFieldList

LOG = logging.getLogger(__name__)


def _lost(f: Any) -> None:
    """Raise a ValueError indicating a lost field.

    Parameters
    ----------
    f : Any
        The lost field.
    """
    raise ValueError(f"Lost field {f}")


def _flatten(params: list[Any] | tuple[Any, ...]) -> list[str]:
    """Flatten a list of parameters.

    Parameters
    ----------
    params : list or tuple of Any
        List or tuple of parameters to flatten.

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


class Grouper(ABC):
    """Group matching fields by some criteria.

    Parameters
    ----------
    params : list of str
        List of parameters to group by.
    """

    def __init__(self, params: list[str]) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = _flatten(params)

    def iterate(self, data: list[Any], *, other: Callable[[Any], None] = _lost) -> Iterator[tuple[Any, ...]]:
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

        groups = self._get_groups(data, other=other)

        for _, group in groups.items():
            if len(group) != len(self.params):
                raise ValueError(f"Missing component. Want {sorted(self.params)}, got {sorted(groups.keys())}")

            yield tuple(group[p] for p in self.params)

    @abstractmethod
    def _get_groups(self, data: list[Any], *, other: Callable[[Any], None] = _lost) -> None: ...

    def _get_param_and_key(self, f):
        key = f.metadata(namespace="mars")
        if not key:
            keys = [k for k in f.metadata().keys() if k not in ("latitudes", "longitudes", "values")]
            key = {k: f.metadata(k) for k in keys}
            if not keys:
                raise NotImplementedError(f"GroupByParam: {f} has no sufficient metadata")

        param = key.pop("param", f.metadata("param"))

        return param, key


class GroupByParam(Grouper):
    """Group matching fields by parameters name.

    Parameters
    ----------
    params : list of str
        List of parameters to group by.
    """

    def _get_groups(self, data: list[Any], *, other: Callable[[Any], None] = _lost) -> None:
        assert callable(other), type(other)

        groups = defaultdict(dict)
        seen = set()

        for f in data:
            param, key = self._get_param_and_key(f)

            if param not in self.params:
                other(f)
                continue

            key = tuple(sorted(key.items()))

            if (param, key) in seen:
                raise ValueError(f"Duplicate component {param} for {key}")
            seen.add((param, key))

            groups[key][param] = f

        return groups


class GroupByParamVertical(Grouper):
    def _get_groups(self, data: list[Any], *, other: Callable[[Any], None] = _lost) -> None:
        assert callable(other), type(other)

        groups = defaultdict(lambda: defaultdict(SimpleFieldList))

        seen = set()

        for f in data:

            param, key = self._get_param_and_key(f)

            if param not in self.params:
                other(f)
                continue

            key.pop("levtype", None)
            level = key.pop("levelist", None)

            key = tuple(sorted(key.items()))

            if (param, level, key) in seen:
                raise ValueError(f"Duplicate component {param} for {key} and level {level}")

            seen.add((param, level, key))

            if level is None:
                groups[key][param] = f
            else:
                groups[key][param].append(f)

        return groups
