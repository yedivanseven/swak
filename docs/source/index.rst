Welcome to swak's documentation!
=================================
*Swiss army knife for functional data-science projects.*

Introduction
------------
This package is a collection of small, modular, and composable building
blocks implementing frequently occurring operations in typical data-science
applications. In abstracting away boiler-plate code, it thus saves time and effort.

- Consolidate all ways to configure your project (command-line arguments,
  environment variables, and config files) with the :doc:`cli` and :doc:`text`
  packages, respectively.
- Wrap the project config into a versatile :doc:`jsonobject`.
- Focus on writing small, configurable, modular, reusable, and testable
  building blocks. Then use the flow controls in :doc:`funcflow` to compose them
  into arbitrarily complex workflows, that are still easy to maintain and to expand.
- Quickly set up projects on Google BigQuery and Google Cloud Storage, and
  efficiently download lots of data in parallel with :doc:`gcp`.
- Build powerful neural-network architectures from the elements in :doc:`pt` and
  train your deep-learning models with early stopping and checkpointing.
  From feature embedding, over feature importance, to repeated residual blocks,
  a broad variety of options is available.
- And much more ...

Installation
------------
- Create a new virtual environment running at least `python 3.12`.
- The easiest way of installing `swak` is from the python package index
  `PyPI <https://pypi.org/project/swak/>`__, where it is hosted. Simply type

  ``pip install swak``

  or treat it like any other python package in your dependency management.
- If you need support for interacting with the Google Cloud Project,
  in particular Google BigQuery and Google CLoud Storage, install
  *extra* dependencies with:

  ``pip install swak[cloud]``

- In order to use the subpackage `swak.pt`, you need to have `PyTorch <https://pytorch.org/>`__ installed.
  Because there is no way of knowing whether you want to run it on CPU only or also on GPU and, if so,
  which version of CUDA (or ROC) you have installed on your machine and how, it is not an explicit
  dependency of `swak`. You will have to install it yourself, *e.g.*, following
  `these instructions <https://pytorch.org/get-started/locally/>`__.
  If you are using `pipenv` for dependency management, you can also have a look at the
  `Pipfile <https://github.com/yedivanseven/swak/blob/main/Pipfile>`__ in the root of the `swak`
  `repository <https://github.com/yedivanseven/swak>`__ and taylor it to your needs. Personally, I go

  ``pipenv sync --categories=cpu``

  for a CPU only installation of PyTorch and

  ``pipenv sync --categories=cuda``

  if I want GPU support.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   dictionary
   funcflow
   text
   cli
   jsonobject
   pd
   cloud
   pt
   misc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
