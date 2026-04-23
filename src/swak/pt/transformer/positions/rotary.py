from typing import Any, Self
import math
import torch as pt
from ...types import Tensor, PosEnc


class Rotary(PosEnc):
    """Rotary positional encodings for multi-head attention in sequence models.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Each vector in the original sequence is expected
        to be of that dimension.
    context: int
        The maximum sequence length that can be processed. Inputs are
        expected to not exceed this size in their next-to-last dimension.
    n_heads: int
        The number of attention heads. Must integer divide `mod_dim` and the
        result must still be and even number.
    device: str or torch.device, optional
        Torch device to first create the rotary positional encodings on.
        Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the rotary positional encodings in.
        Defaults to ``torch.float``.

    Raises
    ------
    ValueError
        If `n_heads` does not integer divide `mod_dim` or if the result is not
        an even number.

    """

    def __init__(
            self,
            mod_dim: int,
            context: int,
            n_heads: int,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **_: Any
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.__context = context
        self.n_heads = self.__compatible(n_heads)
        self.register_buffer(
            'positional_encodings',
            self._precomputed_encodings_for(device, dtype),
            False
        )

    def __compatible(self, n_heads: int) -> int:
        """Validate compatibility of model dimension and number of heads."""
        if self.mod_dim % n_heads != 0:
            tmp = ('Model dimension ({}) must be integer '
                   'divisible by the number of heads ({})!')
            msg = tmp.format(self.mod_dim, n_heads)
            raise ValueError(msg)
        head_dim = self.mod_dim // n_heads
        if head_dim % 2 != 0:
            tmp = 'Head dimension ({}) must be even!'
            msg = tmp.format(head_dim)
            raise ValueError(msg)
        return n_heads

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
        """Device that the rotary positional encodings reside on."""
        return self.positional_encodings.device

    @property
    def dtype(self) -> pt.dtype:
        """Dtype of the rotary positional encodings."""
        return self.positional_encodings.dtype

    @property
    def head_dim(self) -> int:
        """The dimension of each attention head."""
        return self.mod_dim // self.n_heads

    def _precomputed_encodings_for(
            self,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> Tensor:
        """Generate sinusoidal positional encodings for the given context."""
        # Even integer numbers across the embedding/model dimension
        span = pt.arange(
            start=0,
            end=self.head_dim,
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
        divisors = pt.exp(-span * math.log(self.context) / self.head_dim)
        # Arguments of the trigonometric sine and cosine functions.
        angles = positions * divisors
        # Cosine and sine terms of the rotation matrix for a single tensor.
        encodings = pt.empty(
            1,
            self.context,
            self.head_dim // 2,
            2,
            device=device,
            dtype=dtype
        )
        encodings[..., 0] = pt.cos(angles)
        encodings[..., 1] = pt.sin(angles)
        return encodings

    def forward(self, src: Tensor, offset: int = 0) -> Tensor:
        """Apply rotary positional encodings across all heads of the input.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) for all heads. Must be of dimensions
            (..., `n_heads`, `S`, `head_dim`), where the sequence length `S`
            must not exceed `context` and `head_dim` is the `mod_dim` divided
            by `n_heads`.
        offset: int, optional
            Unused. Only for purposes of API compatibility.

        Returns
        -------
        Tensor
            The input sequence(s) with rotary positional encodings applied to
            all heads.

        """
        seq_len = src.size(-2)
        # Reshape to (..., n_heads, seq_len, head_dim / 2, 2) such that
        # elements 1, 3, 5, etc. have index 0 in the last dimension and
        # elements 2, 4, 6, etc. have index 1 in the last dimension
        reshaped = src.unflatten(-1, (-1, 2))
        # Select suitable views on the required cosine and sine terms
        cos = self.positional_encodings[:, :seq_len, :, 0]
        sin = self.positional_encodings[:, :seq_len, :, 1]
        # Apply rotary encodings to even and odd elements separately and
        # interleave them again to restore the original shape:
        # (..., n_heads, seq_len, head_dim)
        return pt.stack([
            reshaped[..., 0] * cos - reshaped[..., 1] * sin,
            reshaped[..., 0] * sin + reshaped[..., 1] * cos
        ], dim=-1).flatten(-2)

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Rotary
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.context,
            self.n_heads,
            self.device,
            self.dtype
        )
