from typing import Any
import yaml
from yaml import Loader
from ..misc import ArgRepr

type Yaml = dict[str, Any] | list[Any]


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
