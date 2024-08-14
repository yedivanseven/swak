from enum import StrEnum
from typing import Any
from google.cloud.storage import Client, Bucket
from google.api_core.retry import Retry
from ...magic import ArgRepr


class Storage(StrEnum):
    """Specify storage class for blobs in a Google Cloud Storage bucket."""
    STANDARD = 'STANDARD'
    NEARLINE = 'NEARLINE'
    COLDLINE = 'COLDLINE'
    ARCHIVE = 'ARCHIVE'


class GcsBucket(ArgRepr):
    """Create a new bucket on Google Cloud Storage.

    Parameters
    ----------
    project: str
        The project to create the bucket in.
    bucket: str
        The name of the bucket to create.
    location: str
        The physical datacenter location to create the dataset in. See the
        Google Cloud Platform `documentation <https://cloud.google.com/storage/
        docs/locations>`__ for options.
    blob_expire_days: int, optional
        Defaults to ``None``. If sets, blobs older than the specified number of
        days will be automatically deleted.
    labels: dict, optional
        Any number of string-valued labels of the bucket. Defaults to none.
    user_project: str, optional
        The project billed for interacting with the bucket. Defaults to the
        `project`
    storage_class: str, optional
        Defaults storage class for blobs in this bucket. Defaults to ``None``,
        which results in "STANDARD".  Use the ``Storage`` enum to specify
        explicitly.
    requester_pays: bool, optional
        Whether the requester will be billed for interacting with the bucket.
        Defaults to ``False``, which means that the (`user_`)`project` will be
        billed.

    Notes
    -----
    There are a lot more options to set, which have been deliberately omitted
    because of the complexity involved.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            project: str,
            bucket: str,
            location: str,
            blob_expire_days: int | None = None,
            labels: dict[str, str] | None = None,
            user_project: str | None = None,
            storage_class: str | None = None,
            requester_pays: bool = False,
    ) -> None:
        self.project = project.strip(' /.')
        self.bucket = bucket.strip(' /.')
        self.location = location.strip().upper()
        self.blob_expire_days = blob_expire_days
        self.labels = {} if labels is None else labels
        if user_project is None:
            self.user_project = self.project
        else:
            self.user_project = user_project.strip(' /.')
        self.storage_class = storage_class
        self.requester_pays = requester_pays
        super().__init__(
            self.project,
            self.bucket,
            self.location,
            blob_expire_days=self.blob_expire_days,
            labels=self.labels,
            user_project=self.user_project,
            storage_class=self.storage_class,
            requester_pays=self.requester_pays
        )

    def __call__(
            self,
            exists_ok: bool = True,
            retry: Retry | None = None,
            timeout: float | tuple[float, float] | None = None,
            **kwargs: Any
    ) -> tuple[Bucket, bool]:
        """Create a new bucket on Google Cloud Storage.

        Parameters
        ----------
        exists_ok: bool, optional
            Whether to raise a ``Conflict`` exception if the targeted bucket
            already exists or not. Defaults to ``True``.
        retry: Retry, optional
            Retry policy for the request. Defaults to ``None``, which disables
            retries. See the Google Cloud Platform `guide
            <https://cloud.google.com/python/docs/reference/storage/1.39.0/
            retry_timeout#configuring-retries>`__ and `reference
            <https://googleapis.dev/python/google-api-core/latest/retry.html>`__
            for options.
        timeout: float, optional
            The number of seconds to wait for the HTTP response to the API call
            before using `retry` or a tuple with separate values for connection
            and request timeouts. Defaults to ``None``, meaning wait forever.
        **kwargs
            Additional keyword arguments are passed to the constructor of the
            Google Storage client (see `documentation <https://cloud.google.
            com/python/docs/reference/storage/latest/google.cloud.storage.
            client.Client#parameters>`__ for options).

        Raises
        ------
        Conflict
            If `exists_ok` is set to ``False`` and the dataset already exists.

        Returns
        -------
        Bucket
            The existing or newly created bucket. If existing, then the bucket
            is returned unchanged, that is, none of the specified options are
            applied.
        bool
            ``True`` if the requested bucket is newly created and ``False``
            if an existing bucket is returned.

        """
        client = Client(self.project, **kwargs)
        bucket = Bucket(client, self.bucket, self.user_project)

        bucket.requester_pays = self.requester_pays
        bucket.storage_class = self.storage_class
        bucket.labels = self.labels
        if self.blob_expire_days:
            bucket.add_lifecycle_delete_rule(age=self.blob_expire_days)

        if bucket.exists() and exists_ok:
            existing = client.get_bucket(bucket, retry=retry, timeout=timeout)
            return existing, False

        bucket.create(
            client,
            self.project,
            self.location,
            retry=retry,
            timeout=timeout
        )
        return bucket, True
