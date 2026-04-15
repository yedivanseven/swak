import sys
from typing import Self
import torch as pt
import torch.nn as ptn
import torch.nn.functional as ptnf
from ...types import Tensor, Attention, PosEnc
from ...blocks import IdentityBlock


class MultiheadedSelfAttention(Attention):
    """Multi-headed self attention with optional (rotary) positional encodings.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Inputs are expected to be of that size in their
        last dimension.
    n_heads: int, optional
        The number of attention heads. Must integer divide `mod_dim` and the
        result must still be and even number. Defaults to 1.
    bias: bool, optional
        Whether to add learnable bias vectors in the projections from
        input to query, key and value and the final out projection.
        Defaults to ``False``.
    dropout: float, optional
        Apply dropout to the attention weights with this probability during
        training. Defaults to 0.1
    pos_enc: PosEnc, optional
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * has a ``context`` attribute specifying the maximum sequence length,
        * processes tensors with dimensions (..., `n_heads`, `S`, `head_dim`),

        where `S` is the sequence length, and `head_dim` is the `mod_dim`
        divided by `n_heads`. If given, it will  be called on queries and keys.
        Typically, this would be an instance of :class:`Rotary` positional
        encodings. Defaults to an instance of :class:`IdentityBlock`,
        which does nothing.
    device: str or torch.device, optional
        Torch device to compute self attention on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to compute self attention in. Defaults to ``torch.float``.

    Raises
    ------
    ValueError
        If `n_heads` does not integer divide `mod_dim`.

    See Also
    --------
    Rotary
    ~swak.pt.blocks.IdentityBlock

    """

    def __init__(
            self,
            mod_dim: int,
            n_heads: int = 1,
            bias: bool = False,
            dropout: float = 0.1,
            pos_enc: PosEnc | None = None,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.n_heads = self.__valid(n_heads)
        self.bias = bias
        self.dropout = dropout
        self.pos_enc = IdentityBlock(mod_dim) if pos_enc is None else pos_enc
        self.pos_enc = self.pos_enc.to(device=device, dtype=dtype)
        scale = pt.tensor(self.head_dim, dtype=dtype, device=device).pow(-0.5)
        self.register_buffer('scale', scale, persistent=False)
        self.qkv = ptn.Linear(
            mod_dim,
            3 * mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out = ptn.Linear(
            mod_dim,
            mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def __valid(self, n_heads: int) -> int:
        """Validate compatibility of model dimension and number of heads."""
        if self.mod_dim % n_heads != 0:
            tmp = ('Model dimension ({}) must be integer '
                   'divisible by the number of heads ({})!')
            msg = tmp.format(self.mod_dim, n_heads)
            raise ValueError(msg)
        return n_heads

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """Device to compute self attention on."""
        return self.qkv.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """Dtype to compute self attention in."""
        return self.qkv.weight.dtype

    @property
    def head_dim(self) -> int:
        """The dimension of each attention head."""
        return self.mod_dim // self.n_heads

    @property
    def has_pos_enc(self) -> bool:
        """Whether a `pos_enc` module was provided at instantiation or not."""
        return not isinstance(self.pos_enc, IdentityBlock)

    @property
    def context(self) -> int:
        """Maximum context length of the positional encodings, if present."""
        if hasattr(self.pos_enc, 'context'):
            return self.pos_enc.context
        return sys.maxsize

    @property
    def _sizes(self) -> tuple[int, int]:
        """Tuple of n_heads and head_dim to reshape intermediate tensors"""
        return self.n_heads, self.head_dim

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool = True
    ) -> Tensor:
        """Forward pass through multi-headed self attention.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of dimensions (..., `S`, `mod_dim`), with
            sequence length `S`.
        mask: Tensor, optional
            Attention mask with a shape broadcastable to the shape of the
            attention weights (..., `S`, `S`). Two types of masks are
            supported: A boolean mask where a value of ``True`` indicates that
            the element *should* take part in attention or a float mask of the
            same dtype as `src` that is added to the product of queries and
            keys, before taking the softmax. In the latter case, a value of
            0.0 (resulting in unchanged attention weights) indicates that an
            element *should* take part in the attention and a value of "-inf"
            (resulting in a zero attention weight) that it should *not*.
            Defaults to ``None``.
        is_causal: bool, optional
            If set to ``True``, inputs are masked with a `S` x `S` lower
            triangular matrix and `mask` is ignored. Default to ``True``.

        Returns
        -------
        Tensor
            The output has the same shape as the input.

        Important
        ---------
        In adhering to the convention of the `scaled_dot_product_attention
        <https://pytorch.org/docs/stable/generated/torch.nn.functional.
        scaled_dot_product_attention.html>`_, the meaning of ``True`` and
        ``False`` (attend to and *not* attend to, respectively) in boolean
        attention masks is exactly the **opposite** of what it means in the
        `MultiheadAttention <https://pytorch.org/docs/stable/generated/torch.
        nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward>`_.
        Therefore, to stay compatible, use float masks!

        """
        query, key, value = self.qkv(src).chunk(3, -1)
        # Reshape from (..., S, mod_dim) to (..., n_heads, S, head_dim)
        query = query.unflatten(-1, self._sizes).transpose(-2, -3)
        key = key.unflatten(-1, self._sizes).transpose(-2, -3)
        value = value.unflatten(-1, self._sizes).transpose(-2, -3)

        attended = ptnf.scaled_dot_product_attention(
            query=self.pos_enc(query),
            key=self.pos_enc(key),
            value=value,
            attn_mask=None if is_causal else mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale
        )
        # Reshape back from (..., n_heads, S, head_dim) to (..., S, mod_dim)
        return self.out(attended.transpose(-2, -3).flatten(-2))

    def reset_parameters(self) -> None:
        """Reset the internal parameters of the projections and pos_enc."""
        self.qkv.reset_parameters()
        self.out.reset_parameters()
        self.pos_enc.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        MultiheadedSelfAttention
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.pos_enc.new(),
            self.device,
            self.dtype
        )
