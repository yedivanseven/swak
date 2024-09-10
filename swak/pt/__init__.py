from importlib.util import find_spec

required = 'google_cloud_bigquery', 'google_cloud_storage', 'pandas_gbq'

if find_spec('torch') is None:
    msg = 'Install the "torch" package to unlock the PyTorch utilities!'
    raise ImportError(msg)
