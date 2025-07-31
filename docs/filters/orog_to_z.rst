########################
 orog_to_z (Reversible)
########################

The ``orog_to_z`` filter converts orography (in metres) to surface
geopotential height (m\ :sup:`2`/s\ :sup:`2`) using the equation:

.. math::

   z &= g \cdot \textrm{orog}\\

Where `g` refers to the `g_gravitational_acceleration` constant. For details please refer to
   `earthkit-meteo
   <https://earthkit-meteo.readthedocs.io/en/latest/_api/meteo/constants/index.html#meteo.constants.g>`_.

This filter must follow a source that provides orography, which is
replaced by surface geopotential height.

.. literalinclude:: yaml/orog_to_z.yaml
   :language: yaml
