###########################
 rodeo_opera_preprocessing
###########################

The ``rodeo_opera_preprocessing`` function applies filtering to the
OPERA Pan-European composites. This preprocessing consists of:

-  Masking of undetected pixels using the ``mask`` (``dm``) variable
-  Clipping of precipitation values to the range ``[0, MAX_TP]``, where
   ``MAX_TP`` is defined at the configuration level
-  Clipping of the quality index to the range ``[0, 1]``

.. literalinclude:: yaml/rodeo_opera_preprocessing.yaml
   :language: yaml

.. note::

   The ``rodeo_opera_preprocessing`` filter was primarily designed to
   work with the 'OPERA Pan-Europeaan' Composites. It's likely these
   filters would be moved into a pluging in the near-future.
