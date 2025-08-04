######################
 rodeo_opera_clipping
######################

The ``rodeo_opera_clipping`` applies a filter to the OPERA Pan-European
composites to clip precipitation between ``[0,'MAX_TP']`` where
``MAX_TP`` is defined at config level and clipping of the quality index
to be between ``[0,1]``. Additional this filter converts tp from `m` to
`mm`.

.. literalinclude:: yaml/rodeo_opera_clipping.yaml
   :language: yaml

.. note::

   The ``rodeo_opera_clipping`` filter was primarily designed to work
   with the 'OPERA Pan-Europeaan' Composites. It's likely these filters
   would be moved into a pluging in the near-future.
