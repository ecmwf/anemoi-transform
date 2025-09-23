# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


class Origin:

    def __init__(self, origins: dict, variables: list[str], dataset_name: str) -> None:
        self.origins = origins
        self.dataset_name = dataset_name

    @staticmethod
    def from_dict(origins: dict, variables: list[str], dataset_name: str) -> "Origin":
        type = origins.get("type", "unknown")
        type = type.title() + "Origin"
        return globals()[type](origins, variables, dataset_name)

    def __repr__(self):
        return self.origins["name"]


class SourceOrigin(Origin):
    pass


class FilterOrigin(Origin):
    pass


class PipeOrigin(Origin):
    def __repr__(self):
        steps = self.origins.get("steps", [])
        return " -> ".join(
            [str(Origin.from_dict(s, self.origins.get("variables", []), self.dataset_name)) for s in steps]
        )


def make_origins(origins: dict, dataset_name: str) -> Origin:
    result = {}
    for variables, origin in origins.items():
        o = Origin.from_dict(origin, variables, dataset_name)
        for variable in variables:
            result[variable] = o

    result = {k: v for k, v in sorted(result.items())}

    return result
