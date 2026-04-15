from typing import Literal, Any
import torch as pt
from ...misc import ArgRepr

type Opt = Literal['tf32', 'ieee']


class Configure(ArgRepr):
    """Set numerical precision optimizations for PyTorch CUDA.

    Parameters
    ----------
    matmul: str, optional
        Whether to use optimized ("tf32") or standardized ("ieee") float32
        arithmetic for matmul operations. Defaults to "tf32".
    cudnn: str, optional
        Whether to use optimized ("tf32") or standardized ("ieee") float32
        arithmetic for neural-network operations. Defaults to "tf32".

    """

    def __init__(self, matmul: Opt = 'tf32', cudnn: Opt = 'tf32') -> None:
        super().__init__(matmul, cudnn)
        self.matmul = matmul
        self.cudnn = cudnn

    def __call__(self, *args: Any, **_: Any) -> Any:
        """Set numerical precision optimizations for PyTorch CUDA.

        Call arguments are simply passed through, kwargs are ignored.

        Returns
        -------
        When called with a single argument, that argument is returned.
        When called with multiple arguments, their tuple is returned.

        """
        pt.backends.cuda.matmul.fp32_precision = self.matmul
        pt.backends.cudnn.fp32_precision = self.cudnn
        return args[0] if len(args) == 1 else args
