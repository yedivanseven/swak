try:
    import torch as pt
except ModuleNotFoundError:
    msg = 'Install the "torch" package to unlock the PyTorch utilities!'
    raise ImportError(msg)

__all__ = ['device']

device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')
pt.set_default_device(device)
pt.set_float32_matmul_precision('medium')

# ToDo: Implement tools for packed & padded sequences
