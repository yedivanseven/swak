import json
from enum import StrEnum
from typing import Any, overload, Literal
from google.cloud.bigquery import Dataset
from google.cloud.exceptions import NotFound
from .clients import Gbq
from .exceptions import GbqError


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
    gbq: Gbq
         An instance of a wrapped GBQ client.
    dataset: str
        The identifier of the dataset to create. Only letters, numbers, and
        underscores are permitted.
    location: str, optional
        The physical datacenter location to create the dataset in. See the
        Google Cloud Platform `documentation <https://docs.cloud.google.com
        bigquery/docs/locations>`__ for options. Defaults to "europe-north1".
    exists_ok: bool, optional
        Whether to quietly return the requested dataset if it exists or raise
        an exception. Defaults to ``False``.
    name: str, optional
        A human-readable name of the dataset. Defaults to the `dataset`.
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
        Any number of string-valued labels of the dataset.
        Defaults to ``None``.
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

    Raises
    ------
    AttributeError
        If `dataset`, `location`, `name`, or `description` are not strings.
    TypeError
        If any of `table_expire_days`, `partition_expire_days`, or
        `max_travel_time_hours` cannot be cast to an integer.
    ValueError
        If any of `table_expire_days`, `partition_expire_days`, or
        `max_travel_time_hours` are less than one and if any of `collation`,
        `rounding` or `billing` are not allowed options.

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
            gbq: Gbq,
            dataset: str,
            location: str = 'europe-north1',
            exists_ok: bool = False,
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
            tags: dict[str, str] | None = None
    ) -> None:
        self.gbq = gbq
        self.dataset = dataset.strip().strip(' /.')
        self.location = location.strip().lower()
        self.exists_ok = bool(exists_ok)
        self.name = self.dataset if name is None else name.strip()
        self.description = None if description is None else description.strip()
        self.table_expire_days = self.__valid(
            table_expire_days,
            'table_expire_days'
        )
        self.partition_expire_days = self.__valid(
            partition_expire_days,
            'partition_expire_days'
        )
        self.labels = {} if labels is None else dict(labels)
        self.access = access
        self.case_sensitive = bool(case_sensitive)
        if collation is None:
            self.collation = None
        else:
            self.collation = str(Collation(collation))
        if rounding is None:
            self.rounding = None
        else:
            self.rounding = str(Rounding(rounding))
        self.max_travel_time_hours = self.__valid(
            max_travel_time_hours,
            'max_travel_time_hours'
        )
        if billing is None:
            self.billing = None
        else:
            self.billing = str(Billing(billing))
        self.tags = {} if tags is None else dict(tags)

    @staticmethod
    @overload
    def to_ms(days: None) -> None:
        ...

    @staticmethod
    @overload
    def to_ms(days: int) -> str:
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
                'projectId': self.gbq.project,
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

    def __call__(self, *_: Any, **__: Any) -> tuple[str, bool]:
        """Create a Google BigQuery dataset in a Google Cloud Platform project.

        If the dataset already exists and `exists_ok` is ``True``, it is
        returned unchanged, that is, none of the specified options are applied.

        Raises
        ------
        GbqError
            If `exists_ok` is set to ``False`` and the dataset already exists.

        Returns
        -------
        str
            The name of the existing or newly created dataset.
        bool
            ``True`` if the requested dataset is newly created and ``False``
            if an existing dataset is returned.

        """
        dataset = Dataset.from_api_repr(self.api_repr)
        client = self.gbq()
        try:
            _ = client.get_dataset(dataset.reference)
        except NotFound:
            exists = False
        else:
            exists = True

        if exists:
            if not self.exists_ok:
                tmp = 'Dataset "{}" already exists in location "{}"!'
                msg = tmp.format(self.dataset, self.location)
                raise GbqError(msg)
        else:
            _ = client.create_dataset(dataset, self.exists_ok)

        return self.dataset, not exists

    @staticmethod
    def __valid(age: Any, name: str) -> int | None:
        """Try to convert time specification to a meaningful integer."""
        if age is None:
            return age
        try:
            as_int = int(age)
        except (TypeError, ValueError) as error:
            cls = type(age).__name__
            tmp = '"{}" must at least be convertible to integer, unlike {}!'
            msg = tmp.format(name, cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format(name, as_int)
            raise ValueError(msg)
        return as_int
