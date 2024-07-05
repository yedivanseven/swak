from setuptools import find_packages
import importlib.metadata as meta

PROJECT = find_packages('../..', exclude=['test*', 'Notebook*'])[0]

project = PROJECT
copyright = '2024, Georg Heimel'
author = 'Georg Heimel'
version = meta.version(PROJECT)
release = version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]
# Hack function sphinx.utils.typing.stringify_annotation to not use
# the __origin__ attribute of ParamSpec and/or TypeVarTuple!
autodoc_typehints = 'none'

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
