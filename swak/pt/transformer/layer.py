import warnings
from typing import Self
import torch as pt
import torch.nn as ptn
from torch.nn import LayerNorm, RMSNorm
from ..types import Tensor, Block, PosEnc, Attention
from ..misc import BlockIdentity

__all__ = ['EncoderLayer']


class EncoderLayer(Block):
    """Encoder layer (i.e., self-attention only) to use in a transformer.

    Parameters
    ----------
    attention: Attention
        A suitably parameterized instance of a self-attention block,
        typically :class:`MultiheadedSelfAttention`
        or :class:`GroupedQuerySelfAttention`
    feed_forward: Block
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `attention`.
    pos_enc: PosEnc, optional
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * has a ``context`` attribute specifying the maximum sequence length,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `attention`. If given, it will be called on the input
        tensor first thing. Typically, this would be an instance of
        ``Sinusoidal`` or ``Learnable`` positional encodings. Defaults to an
        instance of :class:`BlockIdentity`, which does nothing.
    bias: bool, optional
        Whether to use a bias in the ``LayerNorm`` components.
        Defaults to ``True``.
    dropout: float, optional
        Fraction of dropout to apply after self-attention and feed-forward.
        Defaults to 0.1
    norm_cls: type, optional
        Which type of norm to use between (sub-)layers. Must be one of
        ``torch.nn.LayerNorm`` (the default) or ``torch.nn.RMSNorm``.
    norm_first: bool, optional
        Whether to normalize inputs to attention and feed-forward or the sum
        of respective inputs and outputs. Defaults to ``True``.
    eps: float, optional
        Add this value to the denominator in the ``LayerNorm`` components.
        Defaults to 1e-5.
    device: str or device, optional
        Torch device to first create the encoder layer on. Defaults to "cpu".
    dtype: dtype, optional
        Torch dtype to first create the layer in. Defaults to ``torch.float``.

    See Also
    --------
    MultiheadedSelfAttention
    GroupedQuerySelfAttention
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            attention: Attention,
            feed_forward: Block,
            pos_enc: PosEnc | None = None,
            bias: bool = True,
            dropout: float = 0.1,
            norm_cls: type[LayerNorm | RMSNorm] = LayerNorm,
            norm_first: bool = True,
            eps: float = 1e-5,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.attention = attention.to(device=device, dtype=dtype)
        self.feed_forward = feed_forward.to(device=device, dtype=dtype)
        pos_enc = pos_enc or BlockIdentity(attention.mod_dim)
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.bias = bias
        self.dropout = dropout
        self.norm_cls = norm_cls
        self.norm_first = norm_first
        self.eps = eps
        self.drop1 = ptn.Dropout(dropout)
        self.drop2 = ptn.Dropout(dropout)
        self.norm1 = norm_cls(
            attention.mod_dim,
            eps=eps,
            elementwise_affine=True,
            device=device,
            dtype=dtype,
            **self.bias_kwarg
        )
        self.norm2 = norm_cls(
            attention.mod_dim,
            eps=eps,
            elementwise_affine=True,
            device=device,
            dtype=dtype,
            **self.bias_kwarg
        )

    def __check(self, pos_enc: Block) -> Block:
        """Warn if both attention and layer apply positional encodings."""
        we_have_pos_enc = not isinstance(pos_enc, BlockIdentity)
        if we_have_pos_enc and self.attention.has_pos_enc:
            msg = ("Attention and layer both apply positional encodings! "
                   "Hope you know what you're doing ...")
            warnings.warn(msg)
        return pos_enc

    @property
    def bias_kwarg(self) -> dict[str, bool]:
        """Extra keyword 'bias' for LayerNorm components, if requested."""
        return {'bias': self.bias} if self.norm_cls == LayerNorm else {}

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.attention.mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.attention.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.attention.dtype

    @property
    def has_pos_enc(self) -> bool:
        """Whether positional encodings are applied."""
        return (
            self.attention.has_pos_enc or
            not isinstance(self.pos_enc, BlockIdentity)
        )

    @property
    def context(self) -> int:
        """Maximum context length given by the positional encodings."""
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.attention.context)
        return self.attention.context

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool = True,
    ) -> Tensor:
        """Forward pass of one encoder layer (i.e., with self.attention only).

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of dimensions (..., `S`, `D`), with sequence
            length `S` and model dimension `D`.
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
        `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.
        Transformer.html#torch.nn.Transformer.forward>`_.
        Therefore, to stay compatible, use float masks!

        """
        positioned = self.pos_enc(src)
        if self.norm_first:
            attended = self.attention(self.norm1(positioned), mask, is_causal)
            normed = self.norm2(src + self.drop1(attended))
            out = normed + self.drop2(self.feed_forward(normed))
        else:
            attended = self.attention(positioned, mask, is_causal)
            normed = self.norm1(src + self.drop1(attended))
            out = self.norm2(normed + self.drop2(self.feed_forward(normed)))
        return out

    def reset_parameters(self) -> None:
        """Reset all internal parameters of the layer."""
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.pos_enc.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        EncoderLayer
            A fresh, new instance of itself.

        """
        we_have_pos_enc = not isinstance(self.pos_enc, BlockIdentity)
        return self.__class__(
            self.attention.new(),
            self.feed_forward.new(),
            self.pos_enc.new() if we_have_pos_enc else None,
            self.bias,
            self.dropout,
            self.norm_cls,
            self.norm_first,
            self.eps,
            self.device,
            self.dtype
        )
