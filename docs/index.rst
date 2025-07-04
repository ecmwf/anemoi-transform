.. _anemoi-transform:

.. _index-page:

##############################################
 Welcome to `anemoi-transform` documentation!
##############################################

.. warning::

   This documentation is work in progress.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

This package provides a series of data transformation and filering
functions for use by components of the Anemoi framework. Particularly
for those packages which handle incoming raw data, i.e. ``datasets`` and
``inference``.

-  :doc:`installing`

*****************
 Anemoi packages
*****************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`
-  :ref:`anemoi-plugins <anemoi-plugins:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

..
   ..................................................................................

..
   From here defines the TOC in the sidebar, but is not rendered directly on the page.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   overview
   installing
   cli/overview

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Creating filters

   filters/single-field-filters
   filters/matching-filters

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: CLI

   cli/get-grid
   cli/make-regrid-matrix

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Modules

   _api/transform.filters
   _api/transform.grids
   _api/transform.variables
   _api/transform.grouping
