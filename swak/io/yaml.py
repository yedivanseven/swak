from typing import Any
from collections.abc import Mapping
import warnings
import yaml
from yaml import Loader
from ..misc import ArgRepr
from .writer import Writer
from .reader import Reader
from .types import (
    LiteralStorage,
    Yaml,
    Storage,
    Mode,
    NotFound,
    LiteralNotFound
)


class YamlWriter(Writer):
    """Save a dictionary to a YAML file on any of the supported file systems.

    Parameters
    ----------
    path: str
        The absolute path to the YAML file to save the dictionary into.
        May include two or more forward slashes (subdirectories will be
        created) and string placeholders (i.e., pairs of curly brackets)
        that will be interpolated when instances are called.
    storage: str
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the `Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: int, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keywords to the constructor of the file system.
    yaml_kws: dict, optional
        Passed on as keyword arguments to PyYaml`s :func:`dump` function.
        See the `PyYaml documentation <https://pyyaml.org/wiki/
        PyYAMLDocumentation>`_ for options.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `yaml_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if either `storage_kws`
        or `yaml_kws` are not dictionaries.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            yaml_kws: Mapping[str, Any] | None = None
    ) -> None:
        self.yaml_kws = {} if yaml_kws is None else dict(yaml_kws)
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WT,
            chunk_size,
            storage_kws,
            self.yaml_kws
        )

    def __call__(self, yml: Yaml, *parts: Any) -> tuple[()]:
        """Write a dictionary-like object to YAML file the given file system.

        Parameters
        ----------
        yml: dict or list
            The mapping to save as YAML.
        *parts: str
            Fragments that will be interpolated into the `path` given at
            instantiation. Obviously, there must be at least as many as
            there are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        Raises
        ------
        IndexError
            If the `path` given at instantiation has more string placeholders
            that there are `parts`.
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.yml")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        if uri := self._uri_from(*parts):
            with self._managed(uri) as file:
                yaml.dump(yml, file, **self.yaml_kws)
        return ()


class YamlReader(Reader):
    """Read a YAML file from any supported file system.

    Parameters
    ----------
    path: str
        Directory under which the YAML file is located or full path to the
        YAML file. If not fully specified here, it can be completed when
        calling instances.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the `Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    loader: type, optional
        The loader class to use. Defaults to ``Loader``. See the
        `PyYaml documentation <https://pyyaml.org/wiki/PyYAMLDocumentation>`_
        for options.
    not_found: str, optional
        What to do if the specified YAML file is not found. One of "ignore",
        "warn", or "raise". Defaults to "raise". Use the ``NotFound`` enum to
        avoid typos!

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not a float, or if
        `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    Storage
    NotFound

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            loader: type = Loader,
            not_found: LiteralNotFound | NotFound = 'raise'
    ) -> None:
        self.loader = loader
        self.not_found = str(NotFound(not_found))
        super().__init__(
            path,
            storage,
            Mode.RB,
            chunk_size,
            storage_kws,
            loader,
            self.not_found
        )

    def __call__(self, path: str = '') -> Yaml:
        """Read a specific YAML file from the specified file system.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, an empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) to the YAML file to read. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        dict
            The parsed contents of the TOML file.

        """
        uri = self._non_root(path)
        try:
            with self._managed(uri) as file:
                yml = yaml.load(file, self.loader)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File "{}" not found!\nReturning empty YAML.'
                    warnings.warn(msg.format(uri))
                    yml = {}
                case NotFound.IGNORE:
                    yml = {}
                case _:
                    raise error
        return yml


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
