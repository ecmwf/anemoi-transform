##################################
 earthkitfieldlambda (Reversible)
##################################

The ``earthkitfieldlambda`` filter allows you to apply an arbitrary
Python function (either provided inline as a lambda or imported from a
module) to fields selected by parameter name. This enables advanced and
flexible transformations that aren't covered by built-in filters. This
filter must follow a source or filter that provides the necessary
parameter(s) as input. No assumptions are made about physical
quantities, it is entirely user-defined.

For example, you can use it to convert temperatures from Kelvin to
Celsius by subtracting a constant.

.. literalinclude:: yaml/earthkitfieldlambda.yaml
   :language: yaml
