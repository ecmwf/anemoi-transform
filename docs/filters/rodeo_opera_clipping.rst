######################
 rodeo_opera_clipping
######################

The ``rodeo_opera_clipping`` applies a filter to the OPERA Pan-European
composites to clip precipitation between ``[0,
max_total_precipitation]``, where ``max_total_precipitation`` is defined
at the configuration level. If no value is passed a default value
(``MAX_TP``) of 10000 is used. The quality index is also clipped to be
between ``[0,1]``. Additionally this filter converts the
``total_precipitation`` field from `m` to `mm`.

.. literalinclude:: yaml/rodeo_opera_clipping.yaml
   :language: yaml

.. note::

   The ``rodeo_opera_clipping`` filter was primarily designed to work
   with the 'OPERA Pan-Europeaan' Composites. It's likely these filters
   will be moved into a plugin in the near-future.
