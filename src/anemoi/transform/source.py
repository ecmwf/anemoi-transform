# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

from .transform import Transform


class Source(Transform):
    """A source transform that provides data."""

    def backward(self, data: Any) -> None:
        """Raises an error as sources do not support backward transformations.

        Parameters
        ----------
        data : Any
            The input data.

        Raises
        ------
        NotImplementedError
            Always raised as sources do not support backward transformations.
        """
        raise NotImplementedError("Sources do not support backward transformations")
