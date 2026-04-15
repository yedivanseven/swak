from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ...types import Tensor, PosEnc


class Learnable(PosEnc):
    """Learnable positional encodings for transformer-based sequence models.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Inputs are expected to be of that size in their
        last dimension.
    context: int
        The maximum sequence length that can be processed. Inputs are
        expected to not exceed this size in their next-to-last dimension.
    device: str or torch.device, optional
        Torch device to create the learnable positional encodings on.
        Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype of the learnable positional encodings.
        Defaults to ``torch.float``.

    Warning
    -------
    Make sure that the `context` reflects the maximum length of the sequences
    that your model sees at **training** time. In contrast to other types of
    positional encodings, which can reasonably be expected to generalize well
    beyond that during inference, positions that have never been encountered
    during training cannot be encoded at all with ``Learnable``.

    See Also
    --------
    Sinusoidal
    Rotary

    """

    def __init__(
            self,
            mod_dim: int,
            context: int,
            *_: Any,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **__: Any
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.__context = context
        self.positional_encodings = ptn.Parameter(
            pt.empty(1, context, mod_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def context(self) -> int:
        """The maximum sequence length."""
        return self.__context

    @property
    def device(self) -> pt.device:
        """Device that the learnable positional encodings reside on."""
        return self.positional_encodings.device

    @property
    def dtype(self) -> pt.dtype:
        """Dtype of the learnable positional encodings."""
        return self.positional_encodings.dtype

    def forward(self, src: Tensor) -> Tensor:
        """Add learnable positional encodings to a sequence of embeddings.

        Parameters
        ----------
        src: Tensor
            Input sequence(s). Must be of dimensions (..., `S`, `mod_dim`),
            where the sequence length `S` must not exceed `context`.

        Returns
        -------
        Tensor
            The input sequence(s) with positional encodings added.

        """
        return src + self.positional_encodings[:, :src.size(-2), :]

    def reset_parameters(self) -> None:
        """Re-initialize the learnable positional encodings."""
        ptn.init.normal_(self.positional_encodings)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Learnable
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.context,
            device=self.device,
            dtype=self.dtype
        )
