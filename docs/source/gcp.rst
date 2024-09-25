gcp
===

.. automodule:: swak.cloud.gcp


.. autoclass:: swak.cloud.gcp.GbqDataset
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GcsBucket
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GbqQuery
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GbqQuery2GcsParquet
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GcsDir2LocalDir
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GcsParquet2DataFrame
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.GbqQuery2DataFrame
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.cloud.gcp.DataFrame2Gbq
   :members:
   :special-members: __call__
   :show-inheritance:


Enums
-----

.. autoclass:: swak.cloud.gcp.Collation
   :show-inheritance:

   .. autoattribute:: SENSITIVE
      :annotation: = ''

   .. autoattribute:: INSENSITIVE
      :annotation: = und:ci


.. autoclass:: swak.cloud.gcp.Rounding
   :show-inheritance:

   .. autoattribute:: HALF_AWAY
      :annotation: = ROUND_HALF_AWAY_FROM_ZERO

   .. autoattribute:: HALF_EVEN
      :annotation: = ROUND_HALF_EVEN


.. autoclass:: swak.cloud.gcp.Billing
   :show-inheritance:

   .. autoattribute:: PHYSICAL
      :annotation: = PHYSICAL

   .. autoattribute:: LOGICAL
      :annotation: = LOGICAL


.. autoclass:: swak.cloud.gcp.Storage
   :show-inheritance:

   .. autoattribute:: STANDARD
      :annotation: = STANDARD

   .. autoattribute:: NEARLINE
      :annotation: = NEARLINE

   .. autoattribute:: COLDLINE
      :annotation: = COLDLINE

   .. autoattribute:: ARCHIVE
      :annotation: = ARCHIVE


.. autoclass:: swak.cloud.gcp.IfExists
   :show-inheritance:

   .. autoattribute:: FAIL
      :annotation: = fail

   .. autoattribute:: REPLACE
      :annotation: = replace

   .. autoattribute:: APPEND
      :annotation: = append
