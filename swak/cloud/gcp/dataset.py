import json
from enum import StrEnum
from typing import Any, overload, Literal
from google.cloud.bigquery import Client, Dataset
from google.cloud.exceptions import NotFound
from google.api_core.retry import Retry

type LiteralCollation = Literal['', 'und:ci']
type LiteralRounding = Literal['ROUND_HALF_AWAY_FROM_ZERO', 'ROUND_HALF_EVEN']
type LiteralBilling = Literal['PHYSICAL', 'LOGICAL']


class Collation(StrEnum):
    """Specify string sorting in tables in a Google BigQuery dataset."""
    SENSITIVE = ''
    INSENSITIVE = 'und:ci'


class Rounding(StrEnum):
    """Specify rounding in tables in a Google BigQuery dataset."""
    HALF_AWAY = 'ROUND_HALF_AWAY_FROM_ZERO'
    HALF_EVEN = 'ROUND_HALF_EVEN'


class Billing(StrEnum):
    """Specify storage billing mode of tables in a Google BigQuery dataset."""
    PHYSICAL = 'PHYSICAL'
    LOGICAL = 'LOGICAL'


class GbqDataset:
    """Create a new dataset in a Google BigQuery project.

    Parameters
    ----------
    project: str
        The project to create the dataset in.
    dataset: str
        The identifier of the dataset to create. Only letters, numbers, and
        underscored are permitted.
    location: str
        The physical datacenter location to create the dataset in. See the
        Google Cloud Platform `documentation <https://cloud.google.com/
        bigquery/docs/locations>`__ for options.
    name: str, optional
        A human-readable name of the dataset. Defaults to the `dataset` it.
    description: str, optional
        A short description of the dataset. Defaults to ``None``.
    table_expire_days: int, optional
        Number of days after which tables are dropped. Defaults to ``None``,
        which results in tables never being dropped.
    partition_expire_days: int, optional
        Number of days after which partitions of partitioned tables are
        dropped. Defaults to ``None``, which results in partitions never
        being dropped.
    labels: dict, optional
        Any number of string-valued labels of the dataset. Defaults to none.
    access: list of dict, optional
        Fined-grained access rights to the dataset (see the Google Cloud
        Platform `documentation <https://cloud.google.com/bigquery/docs/
        reference/rest/v2/datasets#resource-dataset>`__ for details). If not
        given, defaults access rights are set by Google BigQuery.
    case_sensitive: bool, optional
        Whether dataset and table names should be case-sensitive or not.
        Defaults to `True`.
    collation: str, optional
        Default collation mode for string sorting in string columns of tables.
        Defaults to ``None``, which results in case-sensitive behavior. Use
        the ``Collation`` enum to specify explicitly.
    rounding: str, optional
        Default rounding mode. Defaults to ``None``, which lets Google BigQuery
        choose. Use the ``Rounding`` enum to specify explicitly.
    max_travel_time_hours: int, optional
        Define duration of Google Bigquery's "time travel" window in hours,
        i.e., for how long changes can be rolled back and tables can be
        queried "as of" some previous time. Values can be between 48 and 168
        hours (2 to 7 days). Defaults to 168.
    billing: str, optional
        Default billing mode for tables. Defaults to ``None``, which lets
        Google BigQuery choose. Use the ``Billing`` enum to specify explicitly.
    tags: dict, optional
        Associate globally defined tags with this dataset. Defaults to
        ``None``, which result in no tags to be associated.
    **kwargs
        Additional keyword arguments are passed to the constructor of the
        Google BigQuery ``Client`` (see `documentation <https://cloud.google.
        com/python/docs/reference/bigquery/latest/google.cloud.bigquery.
        client.Client#parameters>`__ for options).

    Note
    ----
    Options for linked and external dataset sources, as well as for encryption
    configuration are deliberately omitted. You probably should not play with
    those without consulting your organization's data engineers.

    See Also
    --------
    Collation
    Rounding
    Billing

    """

    def __init__(
            self,
            project: str,
            dataset: str,
            location: str,
            name: str | None = None,
            description: str | None = None,
            table_expire_days: int | None = None,
            partition_expire_days: int | None = None,
            labels: dict[str, str] | None = None,
            access: list[dict[str, str]] | None = None,
            case_sensitive: bool = True,
            collation: Collation | LiteralCollation | None = None,
            rounding: Rounding | LiteralRounding | None = None,
            max_travel_time_hours: int = 168,
            billing: Billing | LiteralBilling | None = None,
            tags: dict[str, str] | None = None,
            **kwargs: Any
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.dataset = dataset.strip().strip(' /.')
        self.location = location.strip().lower()
        self.name = self.dataset if name is None else name.strip()
        self.description = description
        self.table_expire_days = table_expire_days
        self.partition_expire_days = partition_expire_days
        self.labels = {} if labels is None else labels
        self.access = access
        self.case_sensitive = case_sensitive
        self.collation = collation
        self.rounding = rounding
        self.max_travel_time_hours = max_travel_time_hours
        self.billing = billing
        self.tags = {} if tags is None else tags
        self.kwargs = kwargs

    @staticmethod
    @overload
    def to_ms(days: None) -> None:
        ...

    @staticmethod
    @overload
    def to_ms(days: int) -> int:
        ...

    @staticmethod
    def to_ms(days):
        """Convert integer days to millisecond string for the GCP API call."""
        return None if days is None else f'{int(days * 1000 * 60 * 60 * 24)}'

    @property
    def api_repr(self) -> dict[str, Any]:
        """Payload for the API call to the Google Cloud Platform."""
        return {
            'datasetReference': {
                'projectId': self.project,
                'datasetId': self.dataset
            },
            'friendlyName': self.name,
            'description': self.description,
            'defaultTableExpirationMs': self.to_ms(self.table_expire_days),
            'defaultPartitionExpirationMs': self.to_ms(
                self.partition_expire_days
            ),
            'labels': self.labels,
            'access': self.access,
            'location': self.location,
            'isCaseInsensitive': not self.case_sensitive,
            'defaultCollation': self.collation,
            'defaultRoundingMode': self.rounding,
            'maxTimeTravelHours': f'{self.max_travel_time_hours}',
            'storageBillingModel': self.billing,
            'resourceTags': self.tags
        }

    def __repr__(self) -> str:
        return json.dumps(self.api_repr, indent=4)

    def __call__(
            self,
            exists_ok: bool = True,
            retry: Retry | None = None,
            timeout: float | tuple[float, float] | None = None,
    ) -> tuple[Dataset, bool]:
        """Create a Google BigQuery dataset in a Google Cloud Platform project.

        Parameters
        ----------
        exists_ok: bool, optional
            Whether to raise a ``Conflict`` exception if the targeted dataset
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

        Raises
        ------
        Conflict
            If `exists_ok` is set to ``False`` and the dataset already exists.

        Returns
        -------
        Dataset
            The existing or newly created dataset. If existing, then the
            dataset is returned unchanged, that is, none of the specified
            options are applied.
        bool
            ``True`` if the requested dataset is newly created and ``False``
            if an existing dataset is returned.

        """
        dataset = Dataset.from_api_repr(self.api_repr)
        client = Client(self.project, location=self.location, **self.kwargs)
        try:
            _ = client.get_dataset(dataset.reference, retry, timeout)
        except NotFound:
            exist = False
        else:
            exist = True
        return client.create_dataset(dataset, exists_ok, retry, timeout), exist
