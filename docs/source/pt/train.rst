train
=====

.. automodule:: swak.pt.train


.. autoclass:: swak.pt.train.Trainer
   :members:
   :show-inheritance:


.. autoclass:: swak.pt.train.InMemory
   :members:
   :show-inheritance:
   :inherited-members:


.. autoclass:: swak.pt.train.OnDisk
   :members:
   :show-inheritance:
   :inherited-members:


.. autoclass:: swak.pt.train.StepPrinter
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.EpochPrinter
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.TrainPrinter
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.History
   :members:
   :show-inheritance:



Schedulers
----------

.. autoclass:: swak.pt.train.LinearInverse
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.LinearExponential
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.LinearCosine
   :members:
   :special-members: __call__
   :show-inheritance:



Base classes
------------

.. autoclass:: swak.pt.train.TestDataBase
   :members:
   :show-inheritance:


.. autoclass:: swak.pt.train.TrainDataBase
   :members:
   :inherited-members: n, sample
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.StepCallback
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.EpochCallback
   :members:
   :special-members: __call__
   :show-inheritance:


.. autoclass:: swak.pt.train.TrainCallback
   :members:
   :special-members: __call__
   :show-inheritance:
