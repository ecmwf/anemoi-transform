.. _list-of-filters:

#################
 List of Filters
#################

To know which filters are available in anemoi-transform you can run the
command:

.. code:: bash

   anemoi-transform filters list

   List of Filters: # example output
   ------------------
   - apply-mask
   - clear-step
   - clip
   - convert
   - cos-sin-mean-wave-direction
   - d-to-r
   - ddff-to-uv
   - earthkitfieldlambda
   ...

.. note::

   While the docs focus on the forward transformation of these filters,
   many of them also include a reverse transform. Please refer to the
   code implementation for details. Examples about how to use the
   filters can also be found as part of the unit tests of
   `anemoi-transform`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   earthkitfieldlambda
   empty    
   noop  
   orog_to_z  
   q_to_r  
   r_to_d  
   regrid  
   rename  
   sum  
   uv_to_ddff  
   w_to_wz  
