# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Iterator

from .transform import Transform


class Workflow(Transform):
    """A workflow that applies a series of transformations."""

    def __iter__(self) -> Iterator:
        """Returns an iterator over the transformed data.

        Returns
        -------
        Iterator
            An iterator over the transformed data.
        """
        return iter(self(None))

    def __call__(self, data: Any) -> Any:
        return self.forward(data)
