from typing import Any
from botocore.exceptions import ClientError
from ...misc import ArgRepr
from .clients import S3
from .exceptions import S3Error


class S3Bucket(ArgRepr):
    """Create/retrieve a bucket on/from S3-compatible object storage.

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The (unique!) name of the bucket to create. May include any number of
        string placeholders (i.e., pairs of curly brackets) that will be
        interpolated when instances are called.
    location: str, optional
        The physical datacenter location to create the bucket in. See the
        AWS `documentation <https://docs.aws.amazon.com/
        global-infrastructure/latest/regions/aws-regions.html>`_
        for options. Defaults to "eu-west-1".
    exists_ok: bool, optional
        Whether quietly return the requested bucket if it exists or raise an.
        exception. Defaults to ``False``.
    age: int, optional
        Defaults to ``None``. If set, objects older than the specified number
        of days will be automatically deleted.

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
    S3

    """

    def __init__(
            self,
            s3: S3,
            bucket: str,
            location: str = 'eu-west-1',
            exists_ok: bool = False,
            age: int | None = None
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.location = location.strip().lower()
        self.exists_ok = bool(exists_ok)
        self.age = self.__valid(age)
        super().__init__(
            s3,
            self.bucket,
            self.location,
            self.exists_ok,
            self.age
        )

    @property
    def config(self) -> dict[str, Any]:
        """Minimal configuration used for bucket creation (if required)."""
        # In the default location, no bucket config must be given
        if self.location == 'us-east-1':
            return {}
        return {
            'CreateBucketConfiguration': {
                'LocationConstraint': self.location
            }
        }

    @property
    def lifecycle(self) -> dict[str, list[dict[str, Any]]]:
        """Minimal configuration for adding a life-cycle rule (if required)."""
        return {
            'Rules': [
                {
                    'ID': f'delete-objects-after-{self.age}-days',
                    'Filter': {},
                    'Expiration': {
                        'Days': self.age,
                    },
                    'Status': 'Enabled'
                }
            ]
        }

    def __call__(self, *parts: str) -> tuple[str, bool]:
        """Create/retrieve an S3 bucket with the cached options.

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
        S3Error
            If `exists_ok` is set to ``False`` but the `bucket` already exists.

        """
        bucket = self.bucket.format(*parts).strip(' /.')

        client = self.s3()

        try:
            _ = client.head_bucket(Bucket=bucket)
            bucket_exists = True
        except ClientError:
            bucket_exists = False
            _ = client.create_bucket(
                ACL='private',
                Bucket=bucket,
                **self.config
            )

        if bucket_exists and not self.exists_ok:
            tmp = 'Bucket "{}" already exists in location "{}"!'
            msg = tmp.format(bucket, self.location)
            raise S3Error(msg)

        if self.age:
            client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration=self.lifecycle
            )

        client.close()

        return bucket, not bucket_exists

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
