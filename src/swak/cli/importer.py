from importlib import import_module
from ..misc import ArgRepr
from .exceptions import ImporterError


class Importer(ArgRepr):
    """Programmatically import objects from a module under a top-level package.

    For ease of use and clarity in API, relative imports are not supported.
    Objects are instantiated with where to import from and called with what
    to import.

    Parameters
    ----------
    package: str
        Name of the top-level package to import from. Must not start with dots
        but can contain any number of dots to indicate sub-packages.
    module: str, optional
        The specific module to import objects from. May contain dots to
        indicate that it is located further down within some sub-package.
        Defaults to "steps".

    """

    def __init__(self, package: str, module: str = 'steps') -> None:
        self.package = package.strip().strip(' /.')
        self.module = module.strip().strip(' /.')
        super().__init__(self.package, self.module)

    @property
    def path(self) -> str:
        """Full path specification of (sub-)package and module concatenated."""
        return '.'.join([self.package, self.module])

    def __call__(self, *names: str) -> list:
        """Import any number of objects from the specified package.module.

        Parameters
        ----------
        *names: str
            Name(s) of object(s) to import from ``package.module``.

        Returns
        -------
        list
            Imported objects.

        Raises
        ------
        ImporterError
            When the ``package.module`` is mis-specified, can't be found, or
            when the specified object(s) can't be found in it.

        """
        try:
            location = import_module(self.path)
        except (TypeError, ModuleNotFoundError) as error:
            msg = 'Could not import module "{}"!'
            raise ImporterError(msg.format(self.path)) from error
        imports = []
        for name in names:
            try:
                imported = getattr(location, name)
            except AttributeError as err:
                msg = 'Could not import "{}" from module "{}"!'
                raise ImporterError(msg.format(name, self.path)) from err
            imports.append(imported)
        return imports
