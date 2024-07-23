import os
import pkgutil
import warnings
from typing import Any
from ..magic import ArgRepr
from .misc import NotFound


class TextResourceLoader(ArgRepr):
    """Load text files from a resource directory within a python package.

    Parameters
    ----------
    package: str
        Name of the python package under which the text file is located.
    base_dir: str, optional
        Directory under which the text file is located within the python
        package. May contain any number of forward slashes to access nested
        subdirectories. Defaults to "resources".
    not_found: str, optional
        What to do if the specified file is not found. One of "ignore", "warn",
        or "raise". Defaults to "raise". Use the ``NotFound`` enum
        to avoid typos!
    encoding: str, optional
        Encoding of the text file. Defaults to "utf-8".

    See Also
    --------
    NotFound

    """

    def __init__(
            self,
            package: str,
            base_dir: str = 'resources',
            not_found: str = NotFound.RAISE,
            encoding: str = 'utf-8'
    ) -> None:
        self.package = package.strip(' /')
        self.base_dir = base_dir.strip(' /')
        self.not_found = not_found
        self.encoding = encoding.strip()
        super().__init__(
            self.package,
            self.base_dir,
            str(not_found),
            self.encoding
        )

    def __call__(self, path: str, *args: Any) -> str:
        """Load text file from a directory within the specified python package.

        Parameters
        ----------
        path: str
            Path (including file name) relative to the parent python package.
        *args
            Additional arguments will be interpolated into the joined string
            of `base_dir` and `path` by calling its `format` method. Obviously,
            the number of args must be equal to (or greater than) the total
            number of placeholders in the combined `base_dir` and `path`.

        Returns
        -------
        str
            Decoded contents of the specified text file.

        """
        full_path = os.path.join(self.base_dir, path.strip(' /')).format(*args)
        try:
            content = pkgutil.get_data(self.package, full_path)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty string.'
                    warnings.warn(msg.format(full_path))
                    content = b''
                case NotFound.IGNORE:
                    content = b''
                case _:
                    raise error
        return content.decode(self.encoding)
#
