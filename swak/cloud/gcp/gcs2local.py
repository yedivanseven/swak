import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from pathlib import Path
import shutil
from google.cloud import storage as gcs
from ...misc import ArgRepr


class GcsDir2LocalDir(ArgRepr):
    """Download files from Google Cloud Storage to local directory.

    Parameters
    ----------
    project: str
        Project where the `bucket` and blobs reside.
    bucket: str
        Bucket where the blobs reside.
    prefix: str, optional
        The prefix of the blobs to download. Since it (or part of it)
        can also be provided later, when the callable instance is called, it
        is optional here. Defaults to an empty string.
    base_dir: str, optional
        Absolute path to a base directory on the local filesystem.
        Defaults to "/tmp".
    overwrite: bool, optional
        Whether to silently overwrite local destination directory. Defaults
        to ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to simply return the list of files found in the local
        destination directory, if it and any exists. Defaults to ``False``.
    n_threads: int, optional
        Maximum number of blobs to download in parallel. Defaults to 16.
    chunk_size: int, optional
        Chunk size to read from Google Cloud Storage in one API call in MiB.
        Defaults to 10 MiB.
    **kwargs
        Additional keyword arguments are passed to the constructor of the
        Google Storage ``Client`` (see `documentation <https://cloud.google.
        com/python/docs/reference/storage/latest/google.cloud.storage.
        client.Client#parameters>`__ for options).

    Note
    ----
    Blobs that have any more forward slashes in their name than the
    instantiation and call prefixes combined, i.e., that reside in virtual
    "subdirectories", are ignored and are not downloaded.

    """

    __thread = threading.local()

    def __init__(
            self,
            project: str,
            bucket: str,
            prefix: str = '',
            base_dir: str = '/tmp',
            overwrite: bool = False,
            skip: bool = False,
            n_threads: int = 16,
            chunk_size: int = 10,
            **kwargs: Any
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.bucket = bucket.strip().strip(' /.')
        self.prefix = prefix.strip(' ./') + '/' if prefix.strip(' ./') else ''
        self.base_dir = '/' + base_dir.strip().strip(' /')
        self.overwrite = overwrite
        self.skip = skip
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.kwargs = kwargs
        super().__init__(
            self.project,
            self.bucket,
            self.prefix,
            self.base_dir,
            self.overwrite,
            self.skip,
            self.n_threads,
            self.chunk_size,
            **kwargs
        )

    @property
    def chunk_bytes(self) -> int:
        """Bytes to read from Google Cloud Storage in one API call."""
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    def __call__(self, prefix: str = '') -> list[str]:
        """Download files from Google Cloud Storage to local drive.

        Parameters
        ----------
        prefix: str, optional
            The prefix of the files to download. If given here, it will
            be appended to the `prefix` given at instantiation time.
            Defaults to an empty string.

        Returns
        -------
        list
            A list of the fully resolved file names on the local drive.

        Raises
        ------
        FileExistsError
            If `overwrite` is ``False`` and download into an existing folder
            was attempted.

        """
        prefix = prefix.strip(' ./') + '/' if prefix.strip(' ./') else ''
        local = self.prefix + prefix

        files = self.__files_from(local)

        if not files:
            client = gcs.Client(self.project, **self.kwargs)
            blobs = client.list_blobs(self.bucket, prefix=local or None)
            with ThreadPoolExecutor(
                    self.n_threads,
                    initializer=self.__initializer
            ) as pool:
                downloads = as_completed(
                    pool.submit(self.__download, blob.name, local)
                    for blob in blobs
                    if blob.name.count('/') == local.count('/')
                )
                files = [download.result() for download in downloads]

        return files

    def __files_from(self, local: str) -> list[str]:
        """Return files from local directory if configured and present."""
        path = Path(self.base_dir) / local

        if path.exists() and self.overwrite:
            resolved = str(path.resolve())
            shutil.rmtree(resolved) if path.is_dir() else path.unlink()

        if path.exists() and path.is_dir() and self.skip:
            return [
                str(obj.resolve())
                for obj in path.iterdir()
                if obj.is_file()
            ]

        if path.exists() and path.is_dir() and not any(path.iterdir()):
            return []

        path.mkdir(parents=True)
        return []

    def __initializer(self) -> None:
        """Each thread needs its own client as they cannot be pickled."""
        self.__thread.client = gcs.Client(self.project, **self.kwargs)
        self.__thread.bucket = self.__thread.client.get_bucket(self.bucket)

    def __download(self, name: str, local: str) -> str:
        """Download a single file from Google Cloud Storage to local."""
        blob = self.__thread.bucket.get_blob(name)
        blob.chunk_size = self.chunk_bytes
        file = blob.name.split('/')[-1]
        path = Path(self.base_dir) / local / file
        with path.open('wb') as stream:
            blob.download_to_file(stream, raw_download=True)
        return str(path.resolve())
