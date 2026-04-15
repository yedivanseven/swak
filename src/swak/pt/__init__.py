"""Building blocks and tools to construct, train, and manage PyTorch models.

Cast your data into tensors, embed your features, combine them (optionally
extracting feature importance), and pass them through repeated residual layers
of flexible architecture. Custom loss functions and re-parameterized
distributions allow you to make probabilistic predictions. The training loop
is conveniently abstracted away, allowing for (ready-to-use or custom-made)
callbacks and checkpoints. Finally, trained models can be saved and (re)loaded.

Note
----
The tools in this subpackage are mostly geared towards tabular and sequence
data. Image processing is not currently the main focus.

"""

from importlib.util import find_spec

if find_spec('torch') is None:
    msg = 'Install the "torch" package to unlock the PyTorch utilities!'
    raise ImportError(msg)

# ToDo: Implement tools for packed & padded sequences
