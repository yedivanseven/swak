import os
import tomllib
from typing import Callable, Any
import yaml
from yaml import Loader
from ..magic import ArgRepr

type Toml = dict[str, Any]
type Yaml = dict[str, Any] | list[Any]


class TomlReader(ArgRepr):
    """Light wrapper around the python standard library's ``tomllib.load``.

    Parameters
    ----------
    base_dir: str
        Base directory of the TOML file(s) to read. May contain any number of
        forward slashes to access nested subdirectories.
    parse_float: callable, optional
        Will be called with the string of every TOML float to be decoded.
        Defaults to ``float``.

    """

    def __init__(
            self,
            base_dir: str,
            parse_float: Callable[[str], float] = float
    ) -> None:
        self.base_dir = '/' + base_dir.strip(' /')
        self.parse_float = parse_float
        super().__init__(self.base_dir, parse_float)

    def __call__(self, path: str, *args: Any) -> Toml:
        """Read a specific TOML file.

        Parameters
        ----------
        path: str
            Path (including file name) under the `base_dir` of the TOML file
            to read. May contain any number of forward slashes to access
            nested subdirectories.
        *args
            Additional arguments will be interpolated into the joined string
            of `base_dir` and `path` by calling its `format` method. Obviously,
            the number of args must be equal to (or greater than) the total
            number of placeholders in the combined `base_dir` and `path`.

        Returns
        -------
        dict
            The parsed contents of the TOML file.

        """
        full_path = os.path.join(self.base_dir, path.strip(' /')).format(*args)
        with open(full_path, 'rb') as file:
            toml = tomllib.load(file, parse_float=self.parse_float)
        return toml


class YamlReader(ArgRepr):
    """Light wrapper around pyyaml's ``yaml.load`` function.

    Parameters
    ----------
    base_dir: str
        Base directory of the YAML file(s) to read. May contain any number of
        forward slashes to access nested subdirectories.
    loader: type, optional
        The loader class to use. Defaults to ``Loader``

    """

    def __init__(
            self,
            base_dir: str,
            loader: type = Loader
    ) -> None:
        self.base_dir = '/' + base_dir.strip(' /')
        self.loader = loader
        super().__init__(self.base_dir, loader)

    def __call__(self, path: str, *args: Any) -> Yaml:
        """Read a specific YAML file.

        Parameters
        ----------
        path: str
            Path (including file name) under the `base_dir` of the YAML file
            to read. May contain any number of forward slashes to access
            nested subdirectories.
        *args
            Additional arguments will be interpolated into the joined string
            of `base_dir` and `path` by calling its `format` method. Obviously,
            the number of args must be equal to (or greater than) the total
            number of placeholders in the combined `base_dir` and `path`.

        Returns
        -------
        dict or list
            The parsed contents of the YAML file.

        """
        full_path = os.path.join(self.base_dir, path.strip(' /')).format(*args)
        with open(full_path, 'r') as file:
            yml = yaml.load(file, Loader)
        return yml or {}
