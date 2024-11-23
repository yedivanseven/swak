import tomli_w
from typing import Any
from pathlib import Path
import yaml
from yaml import Dumper
from ..misc import ArgRepr
from .misc import Toml, Yaml


class TomlWriter(ArgRepr):
    """Partial of the ``dump`` function in the ``tomli-w`` package.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save the TOML file to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    overwrite: bool, optional
        What to do if the TOML file to write already exists.
        Defaults to ``False``.
    create: bool, optional
        What to do if the directory where the TOML file should be saved does
        not exist. Defaults to ``False``.
    prune: bool, optional
        Whether to silently drop non-string keys and ``None`` values from the
        dictionary-like object to save as TOML. Defaults to ``False```.
    multiline_strings: bool, optional
        Whether to preserve newline bytes in multiline strings.
        Defaults to ``False``.
    indent: int, optional
        Number of spaces for array indentation. Defaults to 4.
    **kwargs
        Additional keyword arguments will be forwarded python`s builtin
        ``open`` function. The `mode` is determined by the `overwrite`
        argument and is, therefore, dropped from the keyword arguments.

    """

    def __init__(
            self,
            path: str = '',
            overwrite: bool = False,
            create: bool = False,
            prune: bool = False,
            multiline_strings: bool = False,
            indent: int = 4,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.overwrite = overwrite
        self.create = create
        self.prune = prune
        self.multiline_strings = multiline_strings
        self.indent = indent
        if 'mode' in kwargs:
            self.kwargs = (kwargs.pop('mode'), kwargs)[1]
        else:
            self.kwargs = kwargs
        super().__init__(
            self.path,
            overwrite,
            create,
            prune,
            multiline_strings,
            indent,
            **self.kwargs
        )

    @property
    def mode(self) -> str:
        """The mode to open the target file with."""
        return 'wb' if self.overwrite else 'xb'

    @staticmethod
    def __stop_recursion_for(obj: Any) -> bool:
        """Criterion to stop the recursion into a dictionary-like object."""
        try:
            mismatch = len([*obj]) != len(set(obj.keys()))
        except (AttributeError, TypeError):
            return True
        return mismatch or not hasattr(obj, '__getitem__')

    def _pruned(self, toml: Toml) -> Toml:
        """Recursively drop fields with ``None`` values and/or non-string keys.

        Parameters
        ----------
        toml
            The dictionary-like object to drop fields from.

        Returns
        -------
        dict
            Dictionary without non-string keys and ``None`` values.

        """
        if self.__stop_recursion_for(toml):
            return toml
        return {
            key: self._pruned(toml[key])
            for key in toml
            if isinstance(key, str) and toml[key] is not None
        }

    def __call__(self, toml: Toml, *parts: str) -> tuple[()]:
        """Serialize a dictionary-like object and write it to TOML file.

        Parameters
        ----------
        toml: dict
            The dictionary-like object to save as TOML.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(self.mode, **self.kwargs) as file:
            tomli_w.dump(
                self._pruned(toml) if self.prune else toml,
                file,
                multiline_strings=self.multiline_strings,
                indent=self.indent
            )
        return ()


class YamlWriter(ArgRepr):
    """Partial of the ``dump`` function in the ``pyyaml`` package.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save the YAML file to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    overwrite: bool, optional
        What to do if the YAML file to write already exists.
        Defaults to ``False``.
    create: bool, optional
        What to do if the directory where the YAML file should be saved does
        not exist. Defaults to ``False``.
    dumper: type, optional
        The dumper class to use. Defaults to ``Dumper``
    **kwargs
        Additional keyword arguments will be forwarded to the wrapped ``dump``
        method of the ``pyyaml`` package. See its `documentation <https://
        pyyaml.org/wiki/PyYAMLDocumentation>`__ for details.

    """

    def __init__(
            self,
            path: str = '',
            overwrite: bool = False,
            create: bool = False,
            dumper: type = Dumper,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.overwrite = overwrite
        self.create = create
        self.dumper = dumper
        self.kwargs = kwargs
        super().__init__(
            self.path,
            overwrite,
            create,
            dumper,
            **self.kwargs
        )

    @property
    def mode(self) -> str:
        """The mode to open the target file with."""
        return 'wt' if self.overwrite else 'xt'

    def __call__(self, yml: Yaml, *parts: str) -> tuple[()]:
        """Serialize the given object and write it to YAML file.

        Parameters
        ----------
        yml: dict or list
            The object to save as YAML.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(self.mode) as file:
            yaml.dump(yml, file, **self.kwargs)
        return ()
