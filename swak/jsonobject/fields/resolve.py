from pathlib import Path


def resolve(path: str) -> str:
    """Resolve the given path to its full name using pathlib's Path.

    Parameters
    ----------
    path: str
        The path to resolve.

    Returns
    -------
    str
        The resolved path.

    """
    return str(Path(path).resolve())
