.. _single-field-filters:

###############################
 Creating Single Field Filters
###############################

To implement a filter which operates on a single field at a time, users
should subclass the ``SingleFieldFilter``.

**************************
 Defining transformations
**************************

Both forward and backward transformations are supported (allowing for
reversible filters).

-  **Forward Transform (Required)**

      -  Must be implemented by subclasses through ``forward_transform``
         method.
      -  Processes each field individually.
      -  (To create fields from NumPy arrays, the
         ``new_field_from_numpy`` method can be used).

      Example implementation:

      .. code:: python

         def forward_transform(self, field):
             transformed_data = field.to_numpy() * 2
             return self.new_field_from_numpy(transformed_data, template=field)

-  **Backward Transform (Optional)**

      -  Can be implemented through ``backward_transform`` method.
      -  Implementation mirrors that of the ``forward_transform``.
      -  Allows for reversible transformations.
      -  (Should be the inverse operation of the forward transform).

**********************
 Configurable filters
**********************

If the filter is configurable in some way (i.e. additional information
can be provided at creation time), then this can be configured through
the class variables ``required_inputs`` and ``optional_inputs``. This
functionality could be used e.g. for scaling factors, variable names,
etc.

-  **Required Inputs**

      -  Defined through the ``required_inputs`` class variable.
      -  Must be a list or tuple of parameter names.
      -  All required inputs must be provided in the constructor of the
         filter as keyword arguments.

      Example implementation:

      .. code:: python

         class MyFilter(SingleFieldFilter):
             required_inputs = ["scale_factor", "offset"]

-  **Optional Inputs**

      -  Defined through the ``optional_inputs`` class variable.
      -  Provides default values for optional parameters.
      -  Can be overridden through the filter constructor keyword
         arguments.

      Example implementation:

      .. code:: python

         class MyFilter(SingleFieldFilter):
             optional_inputs = {"scale_factor": 1.0, "offset": 0.0}

Parameters passed to the constructor are accessible as attributes within
transform and selection methods, e.g.

.. code:: python

   class ScaleFilter(SingleFieldFilter):
       required_inputs = ["scale_factor"]

       def forward_transform(self, field):
           # self.scale_factor is available (passed in through the constructor kwargs)
           return self.new_field_from_numpy(
               field.to_numpy() * self.scale_factor, template=field
           )


   # Usage
   filter = ScaleFilter(scale_factor=2.0)

************************************************
 Validating inputs and other filter preparation
************************************************

If the user wants to do additional work prior to using the filter, e.g.
validation of the inputs passed in via the filter constructor (those
defined in ``required_inputs``), or loading additional ancillary data
for use in the transformation functions, they can do so by defining a
``prepare_filter`` method.

Example:

.. code:: python

   def prepare_filter(self):
       if self.positive_number < 0:
           raise ValueError("positive_number must be positive")

***********************************
 Transforming only selected fields
***********************************

By default, all fields will be selected (i.e. transformed) by the
filter. If the filter should be applied only to specific fields, users
have the option of defining methods for forward and backward selection.
All fields which are not selected for processing are passed through
unchanged.

Note: If the field metadata is unchanged on transformation, only the
forward selection method needs to be implemented, as it will be reused
for the backward selection.

-  **Forward Select**

      -  Implemented via ``forward_select`` method.
      -  Returns a dictionary specifying which fields to transform.
      -  Fields not matching the selection criteria pass through
         unchanged.

      Example implementation:

      .. code:: python

         def forward_select(self):
             # assuming temperature defined through the required_inputs
             return {"param": self.temperature}

-  **Backward Select**

      -  Implemented via ``backward_select`` method.
      -  By default, uses the same selection as ``forward_select``.
      -  Can be overridden if the backward transformation needs
         different field selection.
      -  Particularly useful when forward transforms modify field
         metadata.
