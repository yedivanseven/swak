misc
====

.. automodule:: swak.misc


.. autoattribute:: swak.misc.DEFAULT_FMT
.. autoattribute:: swak.misc.SHORT_FMT
.. autoattribute:: swak.misc.PID_FMT
.. autoattribute:: swak.misc.RAW_FMT
.. autoattribute:: swak.misc.JSON_FMT


.. autoclass:: swak.misc.StdLogger
   :members:
   :show-inheritance:


.. autoclass:: swak.misc.FileLogger
   :members:
   :show-inheritance:


.. autoclass:: swak.misc.ArgRepr
   :members:
   :inherited-members: _name
   :private-members: _name
   :show-inheritance:


.. autoclass:: swak.misc.IndentRepr
   :members:
   :inherited-members: _name
   :private-members: _name
   :show-inheritance:



Enums
-----

.. autoclass:: swak.misc.NotFound
   :show-inheritance:

   .. autoattribute:: IGNORE
      :annotation: = ignore

   .. autoattribute:: WARN
      :annotation: = warn

   .. autoattribute:: RAISE
      :annotation: = raise


.. autoclass:: swak.misc.Bears
   :show-inheritance:

   .. autoattribute:: PANDAS
      :annotation: = pandas

   .. autoattribute:: POLARS
      :annotation: = polars
