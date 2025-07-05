io
==

.. automodule:: swak.io


.. autoclass:: swak.io.DataFrame2Parquet
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.io.TomlWriter
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.io.YamlWriter
   :members:
   :special-members: __call__
   :show-inheritance:



Base classes
------------

.. autoclass:: swak.io.Writer
   :members:
   :private-members: _uri_from, _managed, _tmp
   :show-inheritance:



Enums
-----

.. autoclass:: swak.io.Storage
   :show-inheritance:

   .. autoattribute:: FILE
      :annotation: = file

   .. autoattribute:: S3
      :annotation: = s3

   .. autoattribute:: GCS
      :annotation: = gcs

   .. autoattribute:: MEMORY
      :annotation: = memory


.. autoclass:: swak.io.Mode
   :show-inheritance:

   .. autoattribute:: WB
      :annotation: = wb

   .. autoattribute:: WT
      :annotation: = wt
