"""Tools to interact with elements of the Google Cloud Project (GCP).

Specifically, data scientists tend to interact mostly with Google's BigQuery
(BQ) data-warehouse solution and the Google Cloud Storage (GCS).

"""

from .clients import Gcs
from .bucket import GcsBucket
from .query import GbqQuery
from .query2gcs import GbqQuery2GcsParquet
from .query2df import GbqQuery2DataFrame
from .df2gbq import IfExists, DataFrame2Gbq
from .dataset import Collation, Rounding, Billing, GbqDataset

__all__ = [
    'Gcs',
    'Collation',
    'Rounding',
    'Billing',
    'GbqDataset',
    'GcsBucket',
    'GbqQuery',
    'GbqQuery2GcsParquet',
    'GbqQuery2DataFrame',
    'IfExists',
    'DataFrame2Gbq'
]

# ToDo: Refactor so that partial clients are created outside classes
