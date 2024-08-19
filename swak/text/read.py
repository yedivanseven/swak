import os
import warnings
import tomllib
from typing import Callable, Any
import yaml
from yaml import Loader
from ..magic import ArgRepr
from .misc import NotFound

type Toml = dict[str, Any]
type Yaml = dict[str, Any] | list[Any]


class TomlReader(ArgRepr):
    """Light wrapper around the python standard library's ``tomllib.load``.

    Parameters
    ----------
    base_dir: str, optional
        Base directory of the TOML file(s) to read. May contain any number of
        forward slashes to access nested subdirectories, or string placeholders
        (i.e., curly brackets) to interpolate later. Defaults to the current
        working directory of the python interpreter.
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
            base_dir: str = '',
            not_found: str = NotFound.RAISE,
            parse_float: Callable[[str], float] = float,
            **kwargs: Any
    ) -> None:
        self.base_dir = '/' + base_dir.strip(' /') if base_dir else os.getcwd()
        self.not_found = not_found
        self.parse_float = parse_float
        if 'mode' in kwargs:
            self.kwargs = (kwargs.pop('mode'), kwargs)[1]
        else:
            self.kwargs = kwargs
        super().__init__(
            self.base_dir,
            str(not_found),
            parse_float,
            **self.kwargs
        )

    def __call__(self, path: str) -> Toml:
        """Read a specific TOML file.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, an empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) under the `base_dir` of the TOML file
            to read. May contain any number of forward slashes to access
            nested subdirectories, or string placeholders (i.e., curly brackets)
            to interpolate with `args`.

        Returns
        -------
        dict
            The parsed contents of the TOML file.

        """
        full_path = self.base_dir + '/' + path.strip(' /')
        try:
            with open(full_path, 'rb', **self.kwargs) as file:
                toml = tomllib.load(file, parse_float=self.parse_float)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty TOML.'
                    warnings.warn(msg.format(full_path))
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
    base_dir: str, optional
        Base directory of the YAML file(s) to read. May contain any number of
        forward slashes to access nested subdirectories, or string placeholders
        (i.e., curly brackets) to interpolate later. Defaults to the current
        working directory of the python interpreter.
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
            base_dir: str = '',
            not_found: str = NotFound.RAISE,
            loader: type = Loader,
            **kwargs: Any
    ) -> None:
        self.base_dir = '/' + base_dir.strip(' /') if base_dir else os.getcwd()
        self.not_found = not_found
        self.loader = loader
        if 'mode' in kwargs:
            self.kwargs = (kwargs.pop('mode'), kwargs)[1]
        else:
            self.kwargs = kwargs
        super().__init__(self.base_dir, str(not_found), loader, **self.kwargs)

    def __call__(self, path: str) -> Yaml:
        """Read a specific YAML file.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, and empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) under the `base_dir` of the YAML file
            to read. May contain any number of forward slashes to access
            nested subdirectories, or string placeholders (i.e., curly brackets)
            to interpolate with `args`.

        Returns
        -------
        dict or list
            The parsed contents of the YAML file.

        """
        full_path = self.base_dir + '/' + path.strip(' /')
        try:
            with open(full_path, 'rb', **self.kwargs) as file:
                yml = yaml.load(file, self.loader)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty YAML.'
                    warnings.warn(msg.format(full_path))
                    yml = {}
                case NotFound.IGNORE:
                    yml = {}
                case _:
                    raise error
        return yml or {}
