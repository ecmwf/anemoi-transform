.. _matching-filters:

###########################
 Creating Matching Filters
###########################

To implement a filter which operates on a set of fields, grouped and
matched by metadata, at a time , users should subclass the
``MatchingFieldsFilter``.

**************************
 Defining transformations
**************************

Both forward and backward transformations are supported (allowing for
reversible filters).

-  Field Matching: Fields are grouped using ``GroupByParam`` (which
   groups based on shared metadata (e.g. time, location, level),
   ignoring the param field) .

-  Transform Logic: The transformation logic is defined in the
   ``forward_transform`` method. The filter can be reversed by defining
   the ``backward_transform`` method.

-  **Components**

-  ``forward_arguments`` / ``backward_arguments``
      Properties that return dictionaries of arguments used for
      transformations.

-  ``forward`` / ``backward``
      Group input fields using GroupByParam, then apply the
      transformation function group-wise.

-  ``_transform()``
      Shared logic for applying any transformation:

      -  Group the input fields.
      -  For each group, apply the transformation function.
      -  Aggregate the results.

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

************************************************
 Validating inputs and other filter preparation
************************************************

-  **matching decorator**:

   The matching decorator is designed to decorate the __init__ method of
   subclasses of MatchingFieldsFilter. It helps configure and initialize
   the filter’s internal argument-matching logic based on the function
   signature of its transformation methods (forward_transform and
   backward_transform).

   The decorator takes the following arguments:

   -  select: The parameter to group by.
   -  forward: The fields to forward.
   -  backward: The fields to backward.

   What it does: When you decorate a subclass’s __init__() method with
   @matching(...), it:

   -  Validates your specified matching config (select, forward,
      backward).

   -  Inspects the __init__ method signature to confirm the listed
      arguments exist.

   -  Sets internal state in the object (_forward_arguments,
      _backward_arguments, etc.) so that the forward() and backward()
      methods know how to group and dispatch data fields to the
      transform logic.

   -  Ensures the filter is marked as “initialized”.

      Below you can find an example of how you would use this decorator
      when writing a new subclass of MatchingFieldsFilter, to tell the
      framework which parameters you're expecting to transform.

      .. code:: python

         class MyFilter(MatchingFieldsFilter):
             @matching(select="param", forward=["a", "b"], backward=["c"])
             def __init__(self, a, b, c):
                 self.a = a
                 self.b = b
                 self.c = c

             def forward_transform(self, a: ekd.Field, b: ekd.Field):
                 ...

             def backward_transform(self, c: ekd.Field):
                 ...

******************************************
 Controlling which fields are transformed
******************************************

The output of a matching filter is an iterator formed by earthkit data
field objects (``ekd.Field``).

The ``forward`` and ``backward`` methods of the filter are used to
control which fields are transformed. To return a field you yield it
from the ``forward`` or ``backward`` methods.

.. code:: python

   def forward(self, field_input1: ekd.Field, field_input2: ekd.Field):
       field_output = field_input1 + field_input2
       yield field_output
       yield field_input1
       yield field_input2
