from typing import Any
from itertools import filterfalse
from ...misc import ArgRepr
from .clients import Gcs
from .exceptions import GcsError


class GcsBucket(ArgRepr):
    """Create/retrieve and configure a bucket on Google Cloud Storage (GCS).

    Parameters
    ----------
    gcs: Gcs
         An instance of a wrapped GCS client.
    bucket: str
        The (unique!) name of the bucket to create. May include any number of
        string placeholders (i.e., pairs of curly brackets) that will be
        interpolated when instances are called.
    location: str, optional
        The physical datacenter location to create the bucket in. See the
        Google Cloud Platform `documentation <https://cloud.google.com/storage/
        docs/locations>`_ for options. Defaults to "EUROPE-NORTH1".
    exists_ok: bool, optional
        Whether quietly return the requested bucket if it exists or raise an.
        exception. Defaults to ``False``.
    age: int, optional
        Defaults to ``None``. If set, blobs older than the specified number of
        days will be automatically deleted.
    user_project: str, optional
        The project billed for interacting with the bucket. Defaults to the
        project carried by the `gcs` client.
    requester_pays: bool, optional
        Whether the requester will be billed for interacting with the bucket.
        Defaults to ``False``, meaning that the (`user_`)`project` pays.
    **kwargs
        Any bucket property to set/change on the created/retrieved bucket.
        See the `GCS docs <https://cloud.google.com/python/docs/reference/
        storage/latest/google.cloud.storage.bucket.Bucket>`_ for all options.

    Raises
    ------
    AttributeError
        If `bucket` or `location` are not strings.
    TypeError
        If `age` cannot be cast to an integer.
    ValueError
        If `age` is less than one.

    See Also
    --------
    Gcs

    """

    def __init__(
            self,
            gcs: Gcs,
            bucket: str,
            location: str = 'EUROPE-NORTH1',
            exists_ok: bool = False,
            age: int | None = None,
            user_project: str | None = None,
            requester_pays: bool = False,
            **kwargs: Any
    ) -> None:
        self.gcs = gcs
        self.bucket = bucket.strip(' /.')
        self.location = location.strip().upper()
        self.exists_ok = bool(exists_ok)
        self.age = self.__valid(age)
        self.user_project = self.__stripped(user_project) or self.gcs.project
        self.requester_pays = bool(requester_pays)
        self.kwargs = kwargs
        super().__init__(
            self.gcs,
            self.bucket,
            self.location,
            self.exists_ok,
            self.age,
            self.user_project,
            self.requester_pays,
            **kwargs
        )

    @property
    def lifecycle(self) -> dict[str, Any]:
        """Minimal configuration for adding a life-cycle rule (if required)."""
        return {'action': {'type': 'Delete'}, 'condition': {'age': self.age}}

    def __call__(self, *parts: str) -> tuple[str, bool]:
        """Create/retrieve and configure a bucket on/from GCS..

        Parameters
        ----------
        *parts: str
            Fragments that will be interpolated into the `bucket` given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `bucket`.

        Returns
        -------
        str
            The name of the existing or newly created bucket.
        bool
            ``True`` if the requested bucket is newly created and ``False``
            if an existing bucket is returned.

        Raises
        ------
        GcsError
            If `exists_ok` is set to ``False`` but the `bucket` already exists
            or if you try to set an invalid bucket property from the `kwargs`.

        """
        bucket = self.bucket.format(*parts).strip(' /.')

        client = self.gcs()
        existing = client.lookup_bucket(bucket)

        if existing:
            if not self.exists_ok:
                tmp = 'Bucket "{}" already exists in location "{}"!'
                msg = tmp.format(existing.name, existing.location)
                raise GcsError(msg)
            created = False
        else:
            client.create_bucket(
                bucket,
                self.requester_pays,
                self.gcs.project,
                self.user_project,
                self.location,
            )
            existing = client.get_bucket(bucket)
            created = True

        if self.age:
            existing_rules = existing.lifecycle_rules or []
            other_rules = filterfalse(self.__is_delete_age, existing_rules)
            existing.lifecycle_rules = [*other_rules, self.lifecycle]

        for key, value in self.kwargs.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
            else:
                msg = '"{}" is not a valid bucket property'
                raise GcsError(msg.format(key))

        existing.patch()

        return bucket, created

    @staticmethod
    def __valid(age: Any) -> int | None:
        """Try to convert age to a meaningful integer."""
        if age is None:
            return age
        try:
            as_int = int(age)
        except (TypeError, ValueError) as error:
            cls = type(age).__name__
            tmp = '"{}" must at least be convertible to integer, unlike {}!'
            msg = tmp.format('age', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('age', as_int)
            raise ValueError(msg)
        return as_int

    @staticmethod
    def __stripped(project: str | None) -> str:
        """None-tolerant string strip for cleaning GCP projects."""
        return '' if project is None else project.strip(' /.')

    @staticmethod
    def __is_delete_age(rule: dict[str, Any]) -> bool:
        """Criterion for filtering out existing delete-age life-cycle rules."""
        is_delete = rule.get('action', {}).get('type') == 'Delete'
        has_age = 'age' in rule.get("condition", {})
        return is_delete and has_age
