import os
import json
from json import JSONDecodeError
from ast import literal_eval
from typing import Any
from ..misc import ArgRepr


class EnvParser(ArgRepr):
    """Parse OS environment variables, preferring prefixed over pure versions.

    Sometimes, environment variables desired for individual use are already
    taken by the operating system or some other system component. In these
    cases, one can resort to prefixing these to avoid conflicts. The present
    class is instantiated with that prefix and will resolve conflicts when
    objects are called, returning the OS environment as a dictionary with
    the values of all variables parsed into python literals.

    Parameters
    ----------
    prefix: str, optional
        Prefix of environment variables that would otherwise be shadowed
        by existing ones. Defaults to empty string.

    """

    def __init__(self, prefix: str = '') -> None:
        super().__init__(prefix)
        self.prefix = prefix.strip()

    def __call__(self, env: dict[str, str] | None = None) -> dict[str, Any]:
        """Parse the OS environment, resolving potentially prefixed variables.

        Parameters
        ----------
        env: dict, optional
            Dictionary to be parsed and resolved. Defaults to ``os.environ``.

        Returns
        -------
        dict
            Environment with prefixed keys removed and the values of their
            non-prefixed counterparts updated accordingly.

        """
        env = os.environ if env is None else env
        prefixed = {}
        original = {}
        for key in env:
            if key.startswith(self.prefix):
                prefixed[key.removeprefix(self.prefix)] = env[key]
            else:
                original[key] = env[key]
        merged = original | prefixed
        return {key: self.__parsed(value) for key, value in merged.items()}

    @staticmethod
    def __parsed(value: str) -> Any:
        """Try to parse (string) environment variables into python objects."""
        try:
            parsed = json.loads(value)
        except (TypeError, JSONDecodeError):
            try:
                parsed = literal_eval(value)
            except (TypeError, ValueError, SyntaxError):
                parsed = value
        return parsed
