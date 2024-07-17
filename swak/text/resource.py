import os
import pkgutil
from ..magic import ArgRepr


class TextResourceLoader(ArgRepr):
    """Load a text file from a directory within a python package.

    The class is initialized with the required parameters and the (callable)
    object can be called repeatedly to load various files.

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

    def __call__(self, path: str) -> str:
        """Load text file from a directory within the specified python package.

        Parameters
        ----------
        path: str
            Path (including file name) relative to the parent python package.

        Returns
        -------
        str
            Contents of the specified text file.

        """
        full_path = os.path.join(self.prefix, path.strip(' /'))
        return pkgutil.get_data(self.package, full_path).decode(self.encoding)
