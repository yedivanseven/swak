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

try:
    import torch as pt
except ModuleNotFoundError as error:
    msg = 'Install the "torch" package to unlock the PyTorch utilities!'
    raise ImportError(msg) from error

__all__ = ['DEVICE']

DEVICE = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')
pt.set_default_device(DEVICE)
if pt.cuda.is_available():
    pt.set_float32_matmul_precision('high')
else:
    pt.set_float32_matmul_precision('medium')

# ToDo: Implement tools for packed & padded sequences
# ToDo: Add positional encodings and, potentially, x-former classes
# ToDo: Homogenize call signature (with "bias", "device", and "dtype"?)
# ToDo: Add "device" and "dtype" properties.
# ToDo: Add troch scripting test to unittests
