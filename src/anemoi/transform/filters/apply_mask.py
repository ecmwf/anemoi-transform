# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np

from . import filter_registry
from .base import SimpleFilter


def masks(var, mask, mask_value = 1, threshold = None):
    """ Masks elements in `var` based on the `mask` array, either for a specific value or a threshold condition. """
    if threshold is not None:
        var[mask > threshold] = np.nan
    else:
        var[mask == mask_value] = np.nan
    
    return var


@filter_registry.register("mask")
class MaskVariable(SimpleFilter):
    """A filter to mask variables using external file."""
    def __init__(
        self,
        *,
        var,
        path_to_mask,
        mask_value = 1,
        threshold=None,
        overwrite=True,
    ):
        self.variable = var
        self.mask = ekd.from_source("file", path_to_mask)[0].to_numpy().astype(bool)
        self.mask_value = mask_value
        self.threshold = threshold
        self.overwrite = overwrite

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.variable,
        )

    def backward(self, data):
        raise NotImplementedError("MaskVariable is not reversible")

    def forward_transform(self, var):
        """Mask variable based on external mask file."""

        variable_masked = masks(var.to_numpy(), self.mask, self.mask_value, self.threshold)
        
        if self.overwrite:
            #overwrite the original variable
            yield self.new_field_from_numpy(variable_masked, template=var, param=self.variable)
        else:
            #create a new variable
            yield self.new_field_from_numpy(variable_masked, template=var, param=self.variable+"_masked")
    
    def backward_transform(self, var):
        raise NotImplementedError("MaskVariable is not reversible")
