# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict


def _lost(f):
    raise ValueError(f"Lost field {f}")


class GroupByMarsParam:
    """Group matching fields by MARS paramters name."""

    def __init__(self, params):
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = params

    def iterate(self, data, *, other=_lost):

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
                raise ValueError("Missing component")

            yield tuple(group[p] for p in self.params)
