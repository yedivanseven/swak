# swak
_Swiss army knife for functional data-science projects._

## 1. Introduction
This package is a collection of small, modular, and composable building
blocks implementing frequently occurring operations in building data-science
applications. It thus saves time in abstracting away boiler-plate code.
* Consolidate all ways to configure your project (command-line arguments,
  environment variables, and config files) with the `cli` and `text`
  packages, respectively.
* Wrap the project config into a versatile `jsonobjects`.
* Focus on writing small, configurable, modular, reusable, and testable 
  building blocks. Then use the flow controls in `funcflow` to compose them
  into arbitrarily complex workflows, that are still easy to maintain and expand.
* Quickly set up projects on Google BigQuery and Google Cloud Storage, and
  efficiently download lots of data in parallel with `cloud.gcp`.
* Build powerful neural-network architectures from the elements in `pt` and
  train your deep-learning models with early stopping and checkpointing.
  From feature embedding, over feature importance, to repeated residual blocks,
  a broad variety of options is available.
* And much more ...

## 2. Installation
* Create a new virtual environment running at least `python 3.12`.
* The easiest way of installing `swak` is from the python package index
[PyPi](https://pypi.org/project/swak/), where it is hosted. Simply type
  ```shell
  pip install swak
  ```
  or treat it like any other python package in your dependency management.
* If you need support for interacting with the Google Cloud Project (GCP),
in particular Google BigQuery (BQ) and Google CLoud Storage (GCS), install
_extra_ dependencies with:
  ```shell
  pip install swak[cloud]
  ```
* In order to use the subpackage `swak.pt`, you need to have [PyTorch](https://pytorch.org/) installed.
Because there is no way of knowing whether you want to run it on CPU only or also on GPU and, if so,
which version of CUDA (or ROC) you have installed on your machine and how, it is not an explicit
dependency of `swak`. You will have to install it yourself, _e.g._, following
[these instructions](https://pytorch.org/get-started/locally/).
If you are using `pipenv` for dependency management, you can also have a look at the
[Pipfile](https://github.com/yedivanseven/swak/blob/main/Pipfile) in the root of the `swak`
[repository](https://github.com/yedivanseven/swak) and taylor it to your needs. Personally, I go
  ```shell
  pipenv sync --categories=cpu
  ```
  for a CPU only installation of PyTorch and
  ```shell
  pipenv sync --categories=cuda
  ```
  if I want GPU support.

## 3. Documentation
The API documentation to `swak` is hosted on [GitHub Pages](https://yedivanseven.github.io/swak/).

## 4. Versioning
The `swak` package uses [semantic versioning](https://semver.org/spec/v2.0.0.html), that is,
`vMAJOR.MINOR.PATCH`.
* Changes that break backwards compatibility or other major milestones will be reflected in
  a change of the `MAJOR` version.
* Added functionality will be reflected by a change in the `MINOR` version.
* Bugfixes, typos, and other small changes will be reflected in a jump of the `PATCH` version.

**Important**: _Prior to major version 1, that is, all versions before `v1.x.y` are to be
considered beta versions and any version jump micht break backwards compatibility!_
