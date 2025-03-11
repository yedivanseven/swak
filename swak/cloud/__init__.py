"""Tools to interact with hosted cloud services.

Current supports elements of the Google Cloud Project (GCP) and Amazon Web
Services (AWS).

"""

from importlib.util import find_spec

required = (
    'google_cloud_bigquery',
    'google_cloud_storage',
    'pandas_gbq',
    'boto3'
)

if any(find_spec(package) for package in required) is None:
    msg = 'Install {} with the [cloud] extra to unlock this subpackage!'
    raise ImportError(msg.format(__package__.split('.')[0]))
