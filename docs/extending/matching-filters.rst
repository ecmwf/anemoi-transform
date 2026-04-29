.. _matching-filters:

###########################
 Creating Matching Filters
###########################

Matching filters operate on groups of related fields at once, matched
by their metadata. To create one, subclass ``MatchingFieldsFilter`` and
define a ``MATCHING`` class attribute along with the transformation
methods described below.

**********************************
 Declaring the matching behaviour
**********************************

Every subclass of ``MatchingFieldsFilter`` must define a ``MATCHING``
class attribute of type ``MatchingSpec``. This tells the framework
which fields the filter expects as inputs, and how to group them.

``MatchingSpec`` accepts the following arguments:

``select``
   The metadata key used to identify fields. Currently only ``"param"``
   is supported.

``forward``
   A tuple of names that identify the fields required by the forward
   transformation (see :ref:`forward-backward-transforms` below).

``backward``
   A tuple of names that identify the fields required by the backward
   transformation. May be left empty if no backward transformation is
   needed.

``return_inputs``
   Controls whether input fields are included in the output alongside
   the transformed fields. Possible values:

   - ``"none"`` (default) -- input fields are not returned.
   - ``"all"`` -- all input fields are returned.
   - A tuple of names -- only the listed input fields are returned.
     The names must be a subset of those in ``forward`` and ``backward``.

   This can also be overridden at instantiation time (see
   :ref:`return-inputs`).

``vertical``
   When ``True``, fields are grouped using ``GroupByParamVertical``
   instead of the default ``GroupByParam``. This causes fields that
   share the same ``param`` name but differ in ``levelist`` to be
   collected together into a single ``ekd.FieldList``, rather than
   being treated as separate fields. Use this when a transformation
   needs access to data across multiple vertical levels at once.

Example
=======

.. code:: python

   from anemoi.transform.filters.fields.matching import MatchingFieldsFilter, MatchingSpec

   class MyFilter(MatchingFieldsFilter):
       MATCHING = MatchingSpec(
           select="param",
           forward=("humidity", "temperature"),
           backward=("relative_humidity", "temperature"),
       )

       def __init__(self, *, humidity, temperature, relative_humidity, return_inputs="none"):
           self.humidity = humidity
           self.temperature = temperature
           self.relative_humidity = relative_humidity
           self.return_inputs = return_inputs
           super().__init__()

       def forward_transform(self, humidity: ekd.Field, temperature: ekd.Field):
           ...
           yield self.new_field_from_numpy(result, template=humidity, param=self.relative_humidity)

       def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field):
           ...
           yield self.new_field_from_numpy(result, template=relative_humidity, param=self.humidity)

Validation
==========

When a subclass of ``MatchingFieldsFilter`` is defined, the framework
automatically validates the class at definition time (via
``__init_subclass__``):

- ``MATCHING`` must be present and be an instance of ``MatchingSpec``.
- The ``__init__`` method must accept keyword arguments for every name
  listed in ``forward`` and ``backward``.
- ``forward_transform`` must accept keyword arguments for every name
  listed in ``forward``.
- ``backward_transform`` must accept keyword arguments for every name
  listed in ``backward``.

If any of these checks fail, a ``TypeError`` or ``ValueError`` is
raised immediately when the class is defined.

.. _forward-backward-transforms:

**************************
 Defining transformations
**************************

Subclasses must implement ``forward_transform`` and may optionally
implement ``backward_transform``. These methods receive fields as
keyword arguments (matching the names in the ``MATCHING`` spec) and
must **yield** one or more ``ekd.Field`` objects.

.. note::

   Do **not** override the ``forward`` or ``backward`` methods directly.
   Those are provided by ``MatchingFieldsFilter`` and handle field
   grouping and dispatching automatically. Implement
   ``forward_transform`` and ``backward_transform`` instead.

Forward transform (required)
============================

``forward_transform`` is an abstract method that must be implemented by
every subclass. It receives the matched fields as keyword arguments and
should yield the transformed output fields.

.. code:: python

   def forward_transform(self, humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
       rh = compute_relative_humidity(humidity.to_numpy(), temperature.to_numpy())
       yield self.new_field_from_numpy(rh, template=humidity, param=self.relative_humidity)

Backward transform (optional)
=============================

``backward_transform`` can be implemented to provide a reverse
transformation, enabling reversible filters. Its signature mirrors that
of ``forward_transform``, but uses the names from the ``backward``
tuple in ``MatchingSpec``.

.. code:: python

   def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> Iterator[ekd.Field]:
       q = compute_specific_humidity(relative_humidity.to_numpy(), temperature.to_numpy())
       yield self.new_field_from_numpy(q, template=relative_humidity, param=self.humidity)

If a backward transform is not provided, calling ``backward`` raises
``NotImplementedError``.

Helper methods
==============

``MatchingFieldsFilter`` provides two helper methods for creating
output fields:

``new_field_from_numpy(array, *, template, **kwargs)``
   Creates a new ``ekd.Field`` from a NumPy array, copying metadata
   from ``template``. Any additional keyword arguments (e.g.
   ``param="r"``) override the corresponding metadata values.

``new_fieldlist_from_list(fields)``
   Creates a new ``ekd.FieldList`` from a list of ``ekd.Field`` objects.

*************************************
 How fields are grouped and matched
*************************************

When ``forward`` or ``backward`` is called on a ``MatchingFieldsFilter``,
the framework:

1. Looks up the ``param`` values that this filter cares about by
   reading the instance attributes whose names match the ``forward``
   (or ``backward``) tuple. For example, if ``forward=("humidity",
   "temperature")`` and the instance has ``self.humidity = "q"`` and
   ``self.temperature = "t"``, the filter will look for fields with
   ``param="q"`` and ``param="t"``.

2. Groups the input ``FieldList`` so that fields sharing the same
   metadata (time, level, location, etc.) -- but differing in ``param``
   -- are collected together.

3. For each group, calls ``forward_transform`` (or
   ``backward_transform``) with the matched fields as keyword
   arguments.

4. Any fields in the input whose ``param`` does not match the filter
   are passed through to the output unchanged.

When ``vertical=True`` in the ``MatchingSpec``, fields sharing the same
``param`` but differing in ``levelist`` are collected into a single
``ekd.FieldList`` and passed as one argument. This is useful for
transformations that need the full vertical profile (e.g. computing
pressure at a height level from model-level data).

.. _return-inputs:

***********************************
 Controlling returned input fields
***********************************

By default (``return_inputs="none"``), only the fields yielded by the
transform method are included in the output. The ``return_inputs``
setting controls whether input fields are also returned:

- ``"none"`` -- only transformed fields are returned.
- ``"all"`` -- all matched input fields are prepended to the output.
- A tuple of names -- only the listed input fields are returned.

``return_inputs`` can be set in two ways:

1. In the ``MatchingSpec`` (sets a class-level default).
2. As an ``__init__`` parameter (overrides the class-level default at
   instantiation time). To enable this, accept a ``return_inputs``
   keyword argument in ``__init__`` and assign it to
   ``self.return_inputs`` *before* calling ``super().__init__()``.

.. code:: python

   class MyFilter(MatchingFieldsFilter):
       MATCHING = MatchingSpec(
           select="param",
           forward=("a", "b"),
           backward=("c",),
       )

       def __init__(self, *, a, b, c, return_inputs="none"):
           self.a = a
           self.b = b
           self.c = c
           self.return_inputs = return_inputs
           super().__init__()

       def forward_transform(self, a: ekd.Field, b: ekd.Field):
           ...

       def backward_transform(self, c: ekd.Field):
           ...

The corresponding YAML recipe can control this at runtime:

.. literalinclude:: yaml/return_inputs.yaml
   :language: yaml
