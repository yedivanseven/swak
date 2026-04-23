from typing import Any, Self
import math
import torch as pt
from ...types import Tensor, PosEnc


class Sinusoidal(PosEnc):
    """Sinusoidal positional encodings for transformer-based sequence models.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Inputs are expected to be of that size in their
        last dimension.
    context: int
        The maximum sequence length that can be processed. Inputs are
        expected to not exceed this size in their next-to-last dimension.
    device: str or torch.device, optional
        Torch device to first create the sinusoidal positional encodings on.
        Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the sinusoidal positional encodings in.
        Defaults to ``torch.float``.

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
        self.register_buffer(
            'positional_encodings',
            self._precomputed_encodings_for(device, dtype),
            False
        )

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
        """Device that the sinusoidal positional encodings reside on."""
        return self.positional_encodings.device

    @property
    def dtype(self) -> pt.dtype:
        """Dtype of the sinusoidal positional encodings."""
        return self.positional_encodings.dtype

    def _precomputed_encodings_for(
            self,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> Tensor:
        """Generate sinusoidal positional encodings for the given context."""
        # Even integer numbers across the embedding/model dimension
        span = pt.arange(
            start=0,
            end=self.mod_dim,
            step=2,
            device=device,
            dtype=dtype
        )
        # Indices of the positions in the sequence.
        positions = pt.arange(
            start=0,
            end=self.context,
            device=device,
            dtype=dtype
        ).unsqueeze(1)
        # Multiplicative factors of position indices in sin/cos arguments
        divisors = pt.exp(-span * math.log(self.context) / self.mod_dim)
        # Arguments of the trigonometric sine and cosine functions.
        angles = positions * divisors
        # Final, additive positional encodings.
        encodings = pt.empty(
            1,
            self.context,
            self.mod_dim,
            device=device,
            dtype=dtype
        )
        encodings[:, :, 0::2] = pt.sin(angles)
        encodings[:, :, 1::2] = pt.cos(angles)
        return encodings

    # ToDo. Make forward accept an offset parameter. Test it!
    def forward(self, src: Tensor) -> Tensor:
        """Add sinusoidal positional encodings to a sequence of embeddings.

        Parameters
        ----------
        src: Tensor
            Input sequence(s). Must be of dimensions (..., `S`, `mod_dim`),
            where the sequence length `S` must not exceed `context`.

        Returns
        -------
        Tensor
            The input sequence(s) with sinusoidal positional encodings added.

        """
        return src + self.positional_encodings[:, :src.size(-2), :]

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Sinusoidal
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.context,
            device=self.device,
            dtype=self.dtype
        )
