pl
==

.. automodule:: swak.pl
   :members:
   :special-members: __call__
   :show-inheritance:



io
--

.. autoclass:: swak.pl.io.Parquet2LazyFrame
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pl.io.LazyFrame2Parquet
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pl.io.LazyStorage
   :show-inheritance:

   .. autoattribute:: FILE
      :annotation: = file

   .. autoattribute:: S3
      :annotation: = s3

   .. autoattribute:: GCS
      :annotation: = gs

   .. autoattribute:: AZURE
      :annotation: = az

   .. autoattribute:: HF
      :annotation: = hf


Base classes
------------

.. autoclass:: swak.pl.io.LazyReader
   :members:
   :private-members: _non_root
   :show-inheritance:


.. autoclass:: swak.pl.io.LazyWriter
   :members:
   :private-members: _uri_from
   :show-inheritance:
