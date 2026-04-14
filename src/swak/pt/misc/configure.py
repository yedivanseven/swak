from typing import Literal, Any
import torch as pt
from ...misc import ArgRepr


class ConfigureTorch(ArgRepr):
    """Set PyTorch default device and numerical precision optimizations.

    Parameters
    ----------
    device: torch.device or str, optional
        The default device to use for torch tensors. Defaults to ``None``,
        which results in "cuda" being picked if available, else "cpu".
    matmul: str, optional
        Whether to use optimized ("tf32") or standardized ("ieee") float32
        arithmetic for matmul operations. Defaults to "tf32".
    cudnn: str, optional
        Whether to use optimized ("tf32") or standardized ("ieee") float32
        arithmetic for neural-network operations. Defaults to "tf32".

    """

    def __init__(
            self,
            device: pt.device | str | None = None,
            matmul: Literal['tf32', 'ieee'] = 'tf32',
            cudnn:  Literal['tf32', 'ieee'] = 'tf32'
    ) -> None:
        self.device = self.default if device is None else pt.device(device)
        self.matmul = matmul
        self.cudnn = cudnn
        super().__init__(self.device, matmul, cudnn)

    @property
    def default(self) -> pt.device:
        """Default device if none is given: CUDA if available else CPU."""
        if pt.cuda.is_available():
            return pt.device('cuda')
        return pt.device('cpu')

    def __call__(self, *args: Any, **_: Any) -> Any:
        """Set PyTorch default device and numerical precision optimizations.

        Call arguments are simply passed through, kwargs are ignored.

        Returns
        -------
        When called with a single argument, that argument is returned.
        When called with multiple arguments, their tuple is returned.

        """
        pt.set_default_device(self.device)
        pt.backends.cuda.matmul.fp32_precision = self.matmul
        pt.backends.cudnn.fp32_precision = self.cudnn
        return args[0] if len(args) == 1 else args
