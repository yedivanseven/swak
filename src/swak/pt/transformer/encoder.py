import warnings
from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Tensors1T, PosEnc
from ..blocks import IdentityBlock
from .layer import EncoderLayer

__all__ = ['Encoder']


class Encoder(PosEnc):
    """Flexible transformer encoder.

    Parameters
    ----------
    layer: EncoderLayer
        A suitably parameterized instance of ``EncoderLayer``.
    n_layers: int, optional
        How often the `layer` is repeated in the transformer stack.
        Must be at least 1, the default.
    pos_enc: PosEnc, optional
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``context`` attribute specifying the maximum sequence length,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `layer`. If given, it will be called on the input
        tensor first thing. Typically, this would be an instance of
        ``Sinusoidal`` or ``Learnable`` positional encodings. Defaults to an
        instance of :class:`IdentityBlock`, which does nothing.
    dropout: float, optional
        Apply dropout to the sum of token embedding and positional encodings
        with this probability during training. Defaults to 0.
    device: str or torch.device, optional
        Torch device to first create the transformer on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the transformer encoder stack in.
        Defaults to ``torch.float``.

    Raises
    ------
    ValueError
        If `n_layers` is less than 1.


    See Also
    --------
    EncoderLayer
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            layer: EncoderLayer,
            n_layers: int = 1,
            pos_enc: PosEnc | None = None,
            dropout: float = 0.0,
            device: pt.device = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.n_layers = self.__valid(n_layers)
        self.layers = ptn.ModuleList([
            layer.new().to(device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ])
        pos_enc = pos_enc or IdentityBlock(layer.mod_dim)
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.dropout = dropout
        self.drop = ptn.Dropout(dropout)
        self.norm = layer.norm1.__class__(
            self.mod_dim,
            eps=layer.norm1.eps,
            elementwise_affine=layer.norm1.elementwise_affine,
            device=device,
            dtype=dtype,
            **layer.bias_kwarg
        ) if layer.norm_first else IdentityBlock(layer.mod_dim)

    @staticmethod
    def __valid(n_layers: int) -> int:
        """Check that the number of layers is at least one."""
        if n_layers < 1:
            msg = 'The transformer must have at least 1 layer, not {}!'
            raise ValueError(msg.format(n_layers))
        return n_layers

    def __check(self, pos_enc: PosEnc) -> PosEnc:
        """Check compatibility of encoder and layer positional encodings."""
        we_have_pos_enc = not isinstance(pos_enc, IdentityBlock)
        if we_have_pos_enc and self.layers[0].has_pos_enc:
            msg = ("Encoder and layer(s) both apply positional encodings! "
                   "Hope you know what you're doing ...")
            warnings.warn(msg)
        if not we_have_pos_enc and not self.layers[0].has_pos_enc:
            msg = ("Either the encoder or the layer(s) typically apply "
                   "positional encodings. But you know what you're doing ...")
            warnings.warn(msg)
        return pos_enc

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.layers[0].mod_dim

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.layers[0].device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.layers[0].dtype

    @property
    def context(self) -> int:
        """Maximum context length permitted by the positional encodings."""
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.layers[0].context)
        return self.layers[0].context

    def forward(
            self,
            src: Tensor,
            attn_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
            is_causal: bool = True
    ) -> Tensors1T:
        """Forward pass through the transformer encoder with optional masking.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of token embedding. Expected dimensions are
            (..., `S`, `D`), with `S` the sequence length and `D` the model
            dimension.
        attn_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of the attention weights (..., `S`, `S`) to be added to the
            product of queries and keys, before taking the softmax. A value of
            0.0 (resulting in unchanged attention weights) indicates that an
            element *should* be attended to and a value of "-inf" (resulting
            in a zero attention weight) that it should *not* be attended to.
            Defaults to ``None``.
        src_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of `src` (..., `S`). A value of 0.0 indicates that an
            element *should* be attended to and a value of "-inf" that it
            should *not* be attended to. Defaults to ``None``.
        is_causal: bool, optional
            If set to ``True``, inputs are masked with a causal `S` x `S`
            triangular matrix (as produced by `generate_square_subsequent_mask
            <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.
            html#torch.nn.Transformer.generate_square_subsequent_mask>`_) and
            both `attn_mask` and `src_mask` are ignored.
            Defaults to ``True``.

        Returns
        -------
        Tensor
            Transformed input with again dimensions (..., `S`, `D`).

        Important
        ---------
        Boolean attention masks are not accepted!

        """
        if is_causal or (attn_mask is None and src_mask is None):
            mask = None
        elif src_mask is None:
            mask = attn_mask
        else:
            # Insert a next-to-last dimension to repeat the src_mask in
            reshaped = src_mask.unsqueeze(-2)
            # Construct the arguments to PyTorch tensors' expand method
            sizes = [-1] * reshaped.dim()
            # src_mask will be repeated sequence-length times in new dimension
            sizes[-2] = src_mask.size(-1)
            # Repeat to form a square mask. Shape is now original +1 dim
            src_mask = reshaped.expand(*sizes)
            # Add repeated and reshaped src_mask to attn_mask if present
            mask = src_mask if attn_mask is None else attn_mask + src_mask

        out = self.drop(self.pos_enc(src))
        for layer in self.layers:
            out = layer(out, mask, is_causal)
        return self.norm(out)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters in all components of the model."""
        for layer in self.layers:
            layer.reset_parameters()
        self.pos_enc.reset_parameters()
        self.norm.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Encoder
            A fresh, new instance of itself.

        """
        we_have_pos_enc = not isinstance(self.pos_enc, IdentityBlock)
        return self.__class__(
            self.layers[0],
            self.n_layers,
            self.pos_enc.new() if we_have_pos_enc else None,
            self.dropout,
            self.device,
            self.dtype
        )
