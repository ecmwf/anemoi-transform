###########################
 rodeo_opera_preprocessing
###########################

The ``rodeo_opera_preprocessing`` function applies filtering to the
OPERA Pan-European composites. This preprocessing consists of:

-  Masking of undetected pixels using the ``mask`` variable

-  Clipping of precipitation values to the range ``[0,
   max_total_precipitation]``, where ``max_total_precipitation`` is
   defined at the configuration level. If no value is passed a default
   value (``MAX_TP``) of 10000 is used.

-  Clipping of the quality index to the range ``[0, 1]``

By default the ``mask`` variable is dropped as part of this filter (the
output field just contains ``total_precipitation`` and ``quality``).
This can be controlled by settings the ``return_mask`` flag from
``False`` to ``True``.

.. literalinclude:: yaml/rodeo_opera_preprocessing.yaml
   :language: yaml

.. note::

   The ``rodeo_opera_preprocessing`` filter was primarily designed to
   work with the 'OPERA Pan-Europeaan' Composites. It's likely these
   filters will be moved into a plugin in the near-future.
