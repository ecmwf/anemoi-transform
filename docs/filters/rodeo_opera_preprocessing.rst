###########################
 rodeo_opera_preprocessing
###########################


The ``rodeo_opera_preprocessing`` applies a filtering to the OPERA Pan-European composites consisting in:
- Masking of undetected pixels using the 'mask' (`dm`) variable
- Clipping of precipitation between 0 and the maximum value given by the 'MAX_TP' defined and clipping of the quality index to be between [0,1]

.. literalinclude:: yaml/rodeo_opera_preprocessing.yaml
   :language: yaml

.. note::

   The ``rodeo_opera_preprocessing`` filter was primarily designed to work with the 'OPERA Pan-Europeaan' Composites.
   It's likely these filters would be moved into a pluging in the near-future.
