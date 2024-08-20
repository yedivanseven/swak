from importlib.util import find_spec

if find_spec('torch') is None:
    msg = 'Install {} with the [torch] extra to unlock this subpackage!'
    raise ImportError(msg.format(__package__.split('.')[0]))
