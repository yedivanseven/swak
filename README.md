# swak
_Swiss army knife for functional data-science projects._

## 1. Introduction

## 2. Installation

### 2.1 Preliminaries
Create a new virtual environment running at least `python 3.12`.

### 2.2 The Base Package
#### 2.2.1 From the Python Package Index (PyPi)
The easiest way of installing `swak` is from the python package index
[PyPi](https://pypi.org/project/swak/), where it is hosted. Simply type
```shell
pip install swak
```
or treat it like any other python package in your dependency management.
#### 2.2.1 From the GitHub repository
...

### 2.3 Cloud support
...

### 2.4 PyTorch support
In order to use the subpackage `swak.pt`, you need to have [PyTorch](https://pytorch.org/) installed.
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
