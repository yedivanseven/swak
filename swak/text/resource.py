import os
import pkgutil
from typing import Any
from ..magic import ArgRepr


class TextResourceLoader(ArgRepr):
    """Load text files from a resource directory within a python package.

    Parameters
    ----------
    package: str
        Name of the python package under which the text file is located.
    prefix: str, optional
        Directory under which the text file is located within the python
        package. May contain any number of slashes to access nested
        subdirectories. Defaults to "resources".
    encoding: str, optional
        Encoding of the text file. Defaults to "utf-8".

    """

    def __init__(
            self,
            package: str,
            prefix: str = 'resources',
            encoding: str = 'utf-8'
    ) -> None:
        self.package = package.strip(' /')
        self.prefix = prefix.strip(' /')
        self.encoding = encoding.strip()
        super().__init__(self.package, self.prefix, self.encoding)

    def __call__(self, path: str, *args: Any) -> str:
        """Load text file from a directory within the specified python package.

        Parameters
        ----------
        path: str
            Path (including file name) relative to the parent python package.
        *args
            Additional arguments will be interpolated into the joined string
            of `prefix` and `path` by calling its `format` method. Obviously,
            the number of args must be equal to the total number of
            placeholders in the combined `prefix` and `path`.

        Returns
        -------
        str
            Decoded contents of the specified text file.

        """

        full_path = os.path.join(self.prefix, path.strip(' /')).format(*args)
        return pkgutil.get_data(self.package, full_path).decode(self.encoding)
