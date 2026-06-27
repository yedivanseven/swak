from typing import Any
from importlib.resources import files
import warnings
from ..misc import ArgRepr
from ..io import NotFound, LiteralNotFound


class TextResourceLoader(ArgRepr):
    """Load text files from a resource directory within a python package.

    Parameters
    ----------
    package: str
        Name of the python package under which the text file is located.
    path: str, optional
        Directory under which the text file is located within the python
        package or full path to the text file. If not fully specified here,
        the path must be completed on calling the instance.
        Defaults to "resources".
    not_found: str, optional
        What to do if the specified file is not found. One of "ignore", "warn",
        or "raise". Defaults to "raise". Use the :class:`NotFound` enum
        to avoid typos!
    encoding: str, optional
        Encoding of the text file. Defaults to "utf-8".

    Raises
    ------
    TypeError
        If either `package` or `path` is not a string.

    See Also
    --------
    ~swak.io.NotFound

    """

    def __init__(
            self,
            package: str,
            path: str = 'resources',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE,
            encoding: str = 'utf-8'
    ) -> None:
        self.package = self.__strip(package, 'package')
        self.path = self.__strip(path, 'path')
        self.not_found = str(NotFound(not_found))
        self.encoding = encoding.strip().lower()
        super().__init__(
            self.package,
            self.path,
            self.not_found,
            self.encoding
        )

    @staticmethod
    def __strip(path: Any, name: str) -> str:
        """Try to normalize the path."""
        try:
            stripped = path.strip(' /.')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            msg = '"{}" must be a string, not {}!'
            raise TypeError(msg.format(name, cls)) from error
        return stripped

    def __call__(self, path: str = '') -> str:
        """Load text file from a directory within the specified python package.

        Parameters
        ----------
        path: str, optional
            Path (including file name) relative to the `path` specified at
            instantiation. Defaults to an empty string, which results in an
            unchanged `path` on calling.

        Returns
        -------
        str
            Decoded contents of the specified text file.

        Raises
        ------
        TypeError
            If `path` is not a string.

        """
        stripped = self.__strip(path, 'path')
        if self.path and stripped:
            full_path = self.path + '/' + stripped
        elif stripped:
            full_path = stripped
        else:
            full_path = self.path
        try:
            content = files(
                self.package
            ).joinpath(
                full_path
            ).read_text(
                encoding=self.encoding
            )
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty string.'
                    warnings.warn(msg.format(full_path))
                    content = ''
                case NotFound.IGNORE:
                    content = ''
                case _:
                    raise error
        return content
