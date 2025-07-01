"""Tools to interact with elements of the Google Cloud Project (GCP).

Specifically, data scientists tend to interact mostly with Google's BigQuery
(BQ) data-warehouse solution and the Google Cloud Storage (GCS).

"""

from .clients import Gcs
from .bucket import GcsBucket
from .query import GbqQuery
from .query2gcs import GbqQuery2GcsParquet
from .gcs2local import GcsDir2LocalDir
from .gcs2df import GcsParquet2DataFrame
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
    'GcsDir2LocalDir',
    'GcsParquet2DataFrame',
    'GbqQuery2DataFrame',
    'IfExists',
    'DataFrame2Gbq'
]

# ToDo: Rewrite with Saver/Loader base class so that file/s3/gcs is handled
# ToDo: Refactor so that partial clients are created outside classes
# ToDo: Add move single file to local
