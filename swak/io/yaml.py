from typing import Any
import yaml
from .writer import Writer, Storage, Mode
from .types import LiteralStorage, Yaml


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
            storage_kws: dict[str, Any] | None = None,
            yaml_kws: dict[str, Any] | None = None
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
