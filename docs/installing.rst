############
 Installing
############

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-transform[...options...]

The options are:

-  ``dev``: install the development dependencies
-  ``all``: install all the dependencies

**************
 Contributing
**************

.. code:: bash

   git clone git@github.com:ecmwf/anemoi-transform.git
   cd anemoi-transform
   pip install .[dev]
   pip install -r docs/requirements.txt

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc
