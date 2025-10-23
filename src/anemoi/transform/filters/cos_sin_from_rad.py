# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Iterator
from typing import Any

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("cos_sin_from_rad")
class CosSinFromRad(MatchingFieldsFilter):
    """A filter to convert any variable in radians to cos() and sin() and back."""

    @matching(
        select="param",
        forward=("param",),
        backward=("cos_param", "sin_param"),
    )
    def __init__(
        self,
        param: str,
        cos_param: str | None = None,
        sin_param: str | None = None,
    ) -> None:
        """Initialize the CosSinFromRad filter.

        Parameters
        ----------
        param : str
            The name of the variable.
        cos_param : str, optional
            The name of the cosine of the variable. Default is to prefix "cos_".
        sin_param : str, optional
            The name of the sine of the variable. Default is to prefix "sin_".
        """

        self.param = param
        self.cos_param = cos_param if cos_param is not None else f"cos_{param}"
        self.sin_param = sin_param if sin_param is not None else f"sin_{param}"

    def forward_transform(
        self,
        param: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert a direction variable to its cosine and sine components.

        Parameters
        ----------
        param : ekd.Field
            The direction field.

        Returns
        -------
        Iterator[ekd.Field]
            Fields of cosine and sine of the direction.
        """
        data = param.to_numpy()
        if (min := data.min()) < -2 * np.pi:
            raise ValueError(f"Param {self.param} is expected in radians in the range [-2pi, pi], but {min=}")
        if (max := data.max()) > 2 * np.pi:
            raise ValueError(f"Param {self.param} is expected in radians in the range [-2pi, pi], but {max=}")

        yield self.new_field_from_numpy(np.cos(data), template=param, param=self.cos_param)
        yield self.new_field_from_numpy(np.sin(data), template=param, param=self.sin_param)

    def backward_transform(
        self,
        cos_param: ekd.Field,
        sin_param: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert cosine and sine components back to direction in radians in the range [-pi, pi).

        Parameters
        ----------
        cos_param : ekd.Field
            The cosine of the direction field.
        sin_param : ekd.Field
            The sine of the  direction field.

        Returns
        -------
        Iterator[ekd.Field]
            Field of the direction.
        """
        direction = np.arctan2(sin_param.to_numpy(), cos_param.to_numpy())

        yield self.new_field_from_numpy(direction, template=cos_param, param=self.param)

    def patch_data_request(self, data_request: dict[str, Any]) -> dict[str, Any]:
        """Modify the data request to include the direction.

        Parameters
        ----------
        data_request : Dict[str, Any]
            The original data request.

        Returns
        -------
        Dict[str, Any]
            The modified data request.
        """

        param = data_request.get("param")
        if param is None:
            return data_request

        if self.cos_param in param or self.sin_param in param:
            data_request["param"] = [p for p in param if p not in (self.cos_param, self.sin_param)]
            data_request["param"].append(self.param)

        return data_request
