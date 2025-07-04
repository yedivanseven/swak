from typing import Any
from collections.abc import Callable
from pathlib import Path
import yaml
import json
import gzip
from ..misc import ArgRepr
from .misc import Yaml


# ToDo: Rewrite with Saver/Loader base class so that file/s3/gcs is handled
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
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.overwrite = overwrite
        self.create = create
        self.kwargs = kwargs
        super().__init__(
            self.path,
            overwrite,
            create,
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


class JsonWriter(ArgRepr):
    """Partial of the ``dump`` function in python's own ``json`` package.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save the JSON file to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    overwrite: bool, optional
        What to do if the JSON file to write already exists.
        Defaults to ``False``.
    create: bool, optional
        What to do if the directory where the JSON file should be saved does
        not exist. Defaults to ``False``.
    gzipped: bool, optional
        Write the JSON to a gzip-compressed JSON file if ``True`` and a plain
        text file if ``False``. If left at ``None``, which is the default,
        file names ending in ".gz" will trigger compression whereas all other
        extensions (if any) will not.
    prune: bool, optional
        If ``True``, keys that are not of a basic type (``str``, ``int``,
        ``float``, ``bool``, ``None``) will be skipped instead of raising a
        ``TypeError``. Defaults to ``False``.
    default: callable, optional
        Called on objects that can’t otherwise be serialized. It should return
        a JSON encodable version of the object or raise a ``TypeError``.
        Defaults to ``None``, which raises a ``TypeError``.
    indent: int or str, optional
        If a positive integer or string, JSON array elements and object members
        will be pretty-printed with that indent level. A positive integer
        indents that many spaces per level. A string (such as "\t") is used to
        indent each level. If zero, negative, or "" (the empty string), only
        newlines are inserted. If ``None`` (the default), the most compact
        representation is used.
    **kwargs
        Additional keyword arguments will be forwarded to the builtin ``open``
        or the ``gzip.open`` function, depending on `gzipped`. The `mode` to
        open the file is hardcoded to with ``"wt"`` or ``"xt"`` (depending on
        `create`) and will, therefore, be dropped from the keyword arguments.

    """

    def __init__(
            self,
            path: str = '',
            overwrite: bool = False,
            create: bool = False,
            gzipped: bool | None = None,
            prune: bool = False,
            default: Callable[[Any], Any] | None = None,
            indent: int | str | None = None,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.overwrite = overwrite
        self.create = create
        self.gzipped = gzipped
        self.prune = prune
        self.default = default
        self.indent = indent
        self.kwargs = (kwargs.pop('mode', ''), kwargs)[1]
        super().__init__(
            self.path,
            overwrite,
            create,
            gzipped,
            prune,
            default,
            indent,
            **self.kwargs
        )

    @property
    def mode(self) -> str:
        """The mode to open the target file with."""
        return 'wt' if self.overwrite else 'xt'

    def __call__(self, obj: Yaml, *parts: str) -> tuple[()]:
        """Serialize the given object and write it to JSON file.

        Parameters
        ----------
        obj: dict or list
            The object to save as JSON.
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
        zipped = path.suffix == '.gz' if self.gzipped is None else self.gzipped
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(
                path, self.mode, **self.kwargs
        ) if zipped else path.open(
                self.mode, **self.kwargs
        ) as file:
            json.dump(
                obj,
                file,
                skipkeys=self.prune,
                ensure_ascii=False,
                indent=self.indent,
                default=self.default
            )
        return ()
