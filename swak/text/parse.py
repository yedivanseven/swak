import tomllib
from typing import Callable, Any
import yaml
from yaml import Loader
from ..magic import ArgRepr

type Toml = dict[str, Any]
type Yaml = dict[str, Any] | list[Any]


class TomlParser(ArgRepr):
    """Light wrapper around the python standard library's ``tomllib.loads``.

    Parameters
    ----------
    parse_float: callable, optional
        Will be called with the string of every TOML float to be decoded.
        Defaults to ``float``.

    """

    def __init__(self, parse_float: Callable[[str], float] = float) -> None:
        self.parse_float = parse_float
        super().__init__(parse_float)

    def __call__(self, toml: str) -> Toml:
        """Parse a specific TOML string.

        Parameters
        ----------
        toml: str
            The string to parse.

        Returns
        -------
        dict
            The result of parsing the TOML string.

        """
        return tomllib.loads(toml, parse_float=self.parse_float)


class YamlParser(ArgRepr):
    """Light wrapper around pyyaml's ``yaml.load`` function.

    Parameters
    ----------
    loader: type, optional
        The loader class to use. Defaults to ``Loader``

    """

    def __init__(self, loader: type = Loader) -> None:
        self.loader = loader
        super().__init__(loader)

    def __call__(self, yml: str) -> Yaml:
        """Parse a specific YAML string.

        Parameters
        ----------
        yml: str
            The YAML string to parse

        Returns
        -------
        dict or list
            The result of parsing the YAML string.

        """
        return yaml.load(yml, self.loader) or {}
