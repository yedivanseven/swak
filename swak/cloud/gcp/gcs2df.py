import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from pandas import DataFrame
import pandas as pd
from google.cloud import storage as gcs
from ...misc import ArgRepr


# ToDo: Add pola-rs support and rename to Gcs2Pandas and Gcs2Polars!
class GcsParquet2DataFrame(ArgRepr):
    """Load parquet files from Google Cloud Storage into a pandas dataframe.

    Parameters
    ----------
    project: str
        Project where the `bucket` and parquet files reside.
    bucket: str
        Bucket where the parquet files reside.
    prefix: str, optional
        The prefix of the parquet files to download. Since it (or part of it)
        can also be provided later, when the callable instance is called, it
        is optional here. Defaults to an empty string.
    n_threads: int, optional
        Maximum number of parquet files to download in parallel.
        Defaults to 16.
    chunk_size: int, optional
        Chunk size to read from Google Cloud Storage in one API call in MiB.
        Defaults to 10 MiB.
    **kwargs
        Additional keyword arguments are passed to the constructor of the
        Google Storage ``Client`` (see `documentation <https://cloud.google.
        com/python/docs/reference/storage/latest/google.cloud.storage.
        client.Client#parameters>`__ for options).

    """

    __thread = threading.local()

    def __init__(
            self,
            project: str,
            bucket: str,
            prefix: str = '',
            n_threads: int = 16,
            chunk_size: int = 10,
            **kwargs: Any
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.bucket = bucket.strip().strip(' /.')
        self.prefix = prefix.strip(' ./') + '/' if prefix.strip(' ./') else ''
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.kwargs = kwargs
        super().__init__(
            self.project,
            self.bucket,
            self.prefix,
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

    def __call__(self, prefix: str = '') -> DataFrame:
        """Load parquet files from Google Cloud Storage into pandas DataFrame.

        Parameters
        ----------
        prefix: str, optional
            The prefix of the parquet files to load. If given here, it will
            be appended to the `prefix` given at instantiation time.
            Defaults to an empty string.

        Returns
        -------
        DataFrame
            Concatenated parquet files.

        """
        prefix = prefix.strip(' ./') + '/' if prefix.strip(' ./') else ''
        remote = self.prefix + prefix

        client = gcs.Client(self.project, **self.kwargs)
        blobs = client.list_blobs(self.bucket, prefix=remote or None)

        with ThreadPoolExecutor(
                self.n_threads,
                initializer=self.__initializer
        ) as pool:
            downloads = as_completed(
                pool.submit(self.__download, blob.name)
                for blob in blobs
                if blob.name.count('/') == remote.count('/')
            )
            results = [download.result() for download in downloads]

        df = pd.concat(results, copy=False) if results else DataFrame()
        df.reset_index(drop=True, inplace=True)
        return df

    def __initializer(self) -> None:
        """Each thread needs its own client as they cannot be pickled."""
        self.__thread.client = gcs.Client(self.project, **self.kwargs)
        self.__thread.bucket = self.__thread.client.get_bucket(self.bucket)

    def __download(self, name: str) -> DataFrame:
        """Read parquet from binary blob opened as file-like object."""
        blob = self.__thread.bucket.get_blob(name)
        with blob.open(
                'rb',
                chunk_size=self.chunk_bytes,
                raw_download=True
        ) as stream:
            df = pd.read_parquet(stream, buffer_size=self.chunk_bytes)
        return df
