import warnings
import tomllib
from typing import Any
from collections.abc import Callable
from pathlib import Path
import yaml
from yaml import Loader
from ..misc import ArgRepr
from .misc import NotFound, Toml, Yaml, LiteralNotFound


class TomlReader(ArgRepr):
    """Light wrapper around the python standard library's ``tomllib.load``.

    Parameters
    ----------
    path: str, optional
        Directory under which the TOML file is located or full path to the
        TOML file. If not fully specified here, the path can be completed or
        changed on calling the instance. Defaults to the current working
        directory of the python interpreter.
    not_found: str, optional
        What to do if the specified TOML file is not found. One of "ignore",
        "warn", or "raise". Defaults to "raise". Use the ``NotFound`` enum to
        avoid typos!
    parse_float: callable, optional
        Will be called with the string of every TOML float to be decoded.
        Defaults to ``float``.
    **kwargs
        Additional keyword arguments will be forwarded python`s builtin
        ``open`` function. Since the `mode` must be ``"rb"``, it is dropped
        from the keyword arguments.

    See Also
    --------
    NotFound

    """

    def __init__(
            self,
            path: str = '',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE,
            parse_float: Callable[[str], float] = float,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.not_found = not_found.strip().lower()
        self.parse_float = parse_float
        if 'mode' in kwargs:
            self.kwargs = (kwargs.pop('mode'), kwargs)[1]
        else:
            self.kwargs = kwargs
        super().__init__(self.path, self.not_found, parse_float, **self.kwargs)

    def __call__(self, path: str = '') -> Toml:
        """Read a specific TOML file.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, an empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) to the TOML file to read. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        dict
            The parsed contents of the TOML file.

        """
        path = Path(self.path) / str(path).strip()
        try:
            with path.open('rb', **self.kwargs) as file:
                toml = tomllib.load(file, parse_float=self.parse_float)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty TOML.'
                    warnings.warn(msg.format(path.resolve()))
                    toml = {}
                case NotFound.IGNORE:
                    toml = {}
                case _:
                    raise error
        return toml


class YamlReader(ArgRepr):
    """Light wrapper around pyyaml's ``yaml.load`` function.

    Parameters
    ----------
    path: str, optional
        Directory under which the YAML file is located or full path to the
        YAML file. If not fully specified here, the path can be completed on
        calling the instance. Defaults to the current working directory of the
        python interpreter.
    not_found: str, optional
        What to do if the specified YAML file is not found. One of "ignore",
        "warn", or "raise". Defaults to "raise". Use the ``NotFound`` enum to
        avoid typos!
    loader: type, optional
        The loader class to use. Defaults to ``Loader``
    **kwargs
        Additional keyword arguments will be forwarded python`s builtin
        ``open`` function. The `mode` is hard-coded to ``"rb"`` and is dropped
        from the keyword arguments.

    See Also
    --------
    NotFound

    """

    def __init__(
            self,
            path: str = '',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE,
            loader: type = Loader,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.not_found = not_found.strip().lower()
        self.loader = loader
        if 'mode' in kwargs:
            self.kwargs = (kwargs.pop('mode'), kwargs)[1]
        else:
            self.kwargs = kwargs
        super().__init__(self.path, self.not_found, loader, **self.kwargs)

    def __call__(self, path: str = '') -> Yaml:
        """Read a specific YAML file.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, and empty dictionary is returned.

        Parameters
        ----------
        path: str, optional
            Path (including file name) to the YAML file to read. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        dict or list
            The parsed contents of the YAML file.

        """
        path = Path(self.path) / str(path).strip()
        try:
            with path.open('rb', **self.kwargs) as file:
                yml = yaml.load(file, self.loader)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty YAML.'
                    warnings.warn(msg.format(path.resolve()))
                    yml = {}
                case NotFound.IGNORE:
                    yml = {}
                case _:
                    raise error
        return yml or {}
