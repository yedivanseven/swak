import pkgutil
import warnings
from ..misc import ArgRepr, NotFound, LiteralNotFound


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
        or "raise". Defaults to "raise". Use the ``NotFound`` enum
        to avoid typos!
    encoding: str, optional
        Encoding of the text file. Defaults to "utf-8".

    See Also
    --------
    ~swak.misc.NotFound

    """

    def __init__(
            self,
            package: str,
            path: str = 'resources',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE,
            encoding: str = 'utf-8'
    ) -> None:
        self.package = package.strip().strip(' /.')
        self.path = path.strip().strip(' /')
        self.not_found = not_found.strip().lower()
        self.encoding = encoding.strip()
        super().__init__(
            self.package,
            self.path,
            self.not_found,
            self.encoding
        )

    def __call__(self, path: str = '') -> str:
        """Load text file from a directory within the specified python package.

        Parameters
        ----------
        path: str, optional
            Path (including file name) relative to the `path` specified at
            instantiation. Defaults to an empty string, which results in an
            unchanged `path` on concatenation.

        Returns
        -------
        str
            Decoded contents of the specified text file.

        """
        path = '/' + path.strip(' /') if path.strip(' /') else ''
        full_path = self.path + path
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
