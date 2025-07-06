from typing import Any
import tomli_w
from .types import Toml, LiteralStorage, Storage, Mode
from .writer import Writer


class TomlWriter(Writer):
    """Save a TOML file to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the file to save the TOML into. May include two
        or more forward slashes (subdirectories will be created) and string
        placeholders (i.e., pairs of curly brackets) that will be interpolated
        when instances are called.
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
    prune: bool, optional
        Whether to silently drop non-string keys and ``None`` values from the
        dictionary-like object to save as TOML. Defaults to ``False```.
    toml_kws: dict, optional
        Passed on as keyword arguments to the :func:`tomli-w.dump` function.
        See the `tomli-w GitHub page <https://github.com/hukkin/tomli-w>`_
        for  options.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `toml_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if either `storage_kws`
        or `toml_kws` are not dictionaries.

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
            prune: bool = False,
            toml_kws: dict[str, Any] | None = None,
    ) -> None:
        self.prune = bool(prune)
        self.toml_kws = {} if toml_kws is None else dict(toml_kws)
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WB,
            chunk_size,
            storage_kws,
            prune,
            self.toml_kws,
        )

    @staticmethod
    def __stop_recursion_for(obj: Any) -> bool:
        """Criterion to stop the recursion into a dictionary-like object."""

        try:
            _ = [*obj]
        except TypeError:
            iterable = False
        else:
            iterable = True
        return not iterable or isinstance(obj, str)

    def _pruned(self, toml: Any) -> Toml:
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
        } if hasattr(toml, 'keys') else [
            self._pruned(item)
            for item in toml
            if item is not None
        ]

    def __call__(self, toml: Toml, *parts: Any) -> tuple[()]:
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

        Raises
        ------
        IndexError
            If the `path` given at instantiation has more string placeholders
            that there are `parts`.
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.toml")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.
        TypeError
            If the dictionary-like object contains ``None`` values and
            `pruned` is ``False``.

        """
        if uri := self._uri_from(*parts):
            pruned = self._pruned(toml) if self.prune else toml
            with self._managed(uri) as file:
                tomli_w.dump(pruned, file, **self.toml_kws)
        return ()
