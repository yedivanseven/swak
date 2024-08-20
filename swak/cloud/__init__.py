from importlib.util import find_spec

required = 'google_cloud_bigquery', 'google_cloud_storage', 'pandas_gbq'

if any(find_spec(package) for package in required) is None:
    msg = 'Install {} with the [cloud] extra to unlock this subpackage!'
    raise ImportError(msg.format(__package__.split('.')[0]))
