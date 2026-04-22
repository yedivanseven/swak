import warnings
from typing import Self, Any
import torch as pt
import torch.nn as ptn
from torch.nn import LayerNorm, RMSNorm
from ..blocks import IdentityBlock
from ..types import Tensor, PosEnc, Resettable, Trafo, Block, Attn

type Tensors2T = tuple[Tensor, Tensor | None]
type Tensors3T = tuple[Tensor | None, Tensor | None, Tensor | None]


class Compressor(Trafo):
    """Context-aware length-compression wrapper around sequence models.

    The :class:`Compressor` uses self- and cross-attention to compress
    incoming sequences of embeddings in length by a fixed factor of 2.
    The compressed sequences are then sent through the wrapped model and
    inflated again to the original length using cross-attention with U-net
    style residual connections to the compression stage.

    Parameters
    ----------
    model: Resettable
        The sequence model to wrap, typically an :class:`Encoder`. It will be
        called with 3 arguments: the compressed sequences, a correspondingly
        compressed attention mask (or ``None``), and a boolean flag indicating
        whether the attention mask is causal or not.
    attend: Attn
        A suitably parameterized instance of a self-attention block,
        typically :class:`MultiheadedSelfAttention`
        or :class:`GroupedQuerySelfAttention`.
    forward: Block
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `attention`.
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
    bias: bool, optional
        Whether to use a bias in the cross-attention components.
        Defaults to ``False``.
    dropout: float, optional
        Apply this amount of dropout to the sum of embeddings and positional
        encodings as well as to the outputs of each sub-layer. Defaults to 0.
    norm_first: bool, optional
        Whether to normalize inputs to attentions and feed-forwards or the sum
        of respective inputs and outputs. Defaults to ``True``.
    norm_cls: type, optional
        Which type of norm to use between (sub-)layers. Must be one of
        ``torch.nn.LayerNorm`` (the default) or ``torch.nn.RMSNorm``.
    *args
        Arguments used to initialize an instance of `norm_cls`.
    device: str or torch.device, optional
        Torch device to first create the compressor on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the compressor in.
        Defaults to ``torch.float``.
    **kwargs
        Keyword arguments used to initialize an instance of `norm_cls`.

    Important
    ---------
    If `norm_first` is ``True``, the wrapped `model` will receive an un-normed
    input and **may** return one. If, however, `norm_first` is ``False``, then
    the wrapped `model` will receive a normed input an **must** return one!

    See Also
    --------
    Encoder
    MultiheadedSelfAttention
    GroupedQuerySelfAttention
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            model: Resettable,
            attend: Attn,
            forward: Block,
            pos_enc: PosEnc | None = None,
            bias: bool = True,
            dropout: float = 0.0,
            norm_first: bool = True,
            norm_cls: type[LayerNorm | RMSNorm] = LayerNorm,
            *args: Any,
            device: pt.device = 'cpu',
            dtype: pt.dtype = pt.float,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.model = model
        self.attend_inp = attend.new().to(device=device, dtype=dtype)
        self.attend_out = attend.new().to(device=device, dtype=dtype)
        self.forward_inp = forward.new().to(device=device, dtype=dtype)
        self.forward_out = forward.new().to(device=device, dtype=dtype)
        pos_enc = pos_enc or IdentityBlock(attend.mod_dim)
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.bias = bias
        self.dropout = dropout
        self.args = args
        self.norm_first = norm_first
        self.norm_cls = norm_cls
        self.kwargs = kwargs
        self.drop = ptn.Dropout(dropout)
        self.compress = ptn.MultiheadAttention(
             embed_dim=self.mod_dim,
             num_heads=self.n_heads,
             dropout=dropout,
             bias=bias,
             batch_first=True,
             device=device,
             dtype=dtype
        )
        self.norm_self_attn_inp = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.norm_cross_attn_inp = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.norm_fwd_inp = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.inflate = ptn.MultiheadAttention(
             embed_dim=self.mod_dim,
             num_heads=self.n_heads,
             dropout=dropout,
             bias=bias,
             batch_first=True,
             device=device,
             dtype=dtype
        )
        self.norm_self_attn_out = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.norm_cross_attn_out = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.norm_fwd_out = norm_cls(
            self.mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )

    def __check(self, pos_enc: PosEnc) -> PosEnc:
        """Check compatibility of encoder and layer positional encodings."""
        we_have_pos_enc = not isinstance(pos_enc, IdentityBlock)
        if we_have_pos_enc and self.attend_inp.has_pos_enc:
            msg = ("Compressor and attention both apply positional encodings!"
                   " Hope you know what you're doing ...")
            warnings.warn(msg)
        if not we_have_pos_enc and not self.attend_inp.has_pos_enc:
            msg = ("Either the compressor or the attention typically apply "
                   "positional encodings. But you know what you're doing ...")
            warnings.warn(msg)
        return pos_enc

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.attend_inp.mod_dim

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.attend_inp.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.attend_inp.dtype

    @property
    def context(self) -> int:
        """Maximum context length permitted by the positional encodings."""
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.attend_inp.context)
        return self.attend_inp.context

    @property
    def has_pos_enc(self) -> bool:
        """Whether positional encodings are applied."""
        return (
            self.attend_inp.has_pos_enc or
            not isinstance(self.pos_enc, IdentityBlock)
        )

    @property
    def n_heads(self) -> int:
        """The number of attention heads used."""
        return self.attend_inp.n_heads

    @staticmethod
    def merge_masks(
            attn_mask: Tensor | None,
            src_mask: Tensor | None,
            is_causal: bool
    ) -> Tensor | None:
        """Utility method to merge attention and source masks if necessary.

        Parameters
        ----------
        attn_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of the attention weights (..., `S`, `S`) to be added to the
            product of queries and keys, before taking the softmax. A value of
            0.0 (resulting in unchanged attention weights) indicates that an
            element *should* be attended to and a value of "-inf" (resulting
            in a zero attention weight) that it should *not* be attended to.
        src_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of `src` (..., `S`). A value of 0.0 indicates that an
            element *should* be attended to and a value of "-inf" that it
            should *not* be attended to.
        is_causal: bool, optional
            If ``True``, inputs are masked with a causal `S` x `S` triangular
            matrix and both `attn_mask` and `src_mask` are ignored.

        Returns
        -------
        Tensor or None
            The merged masks, or ``None`` if none are provided or `is_causal`
            is ``True``.

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
        return mask

    @staticmethod
    def _pad(src: Tensor, mask: Tensor | None) -> Tensors2T:
        """Pad odd-length sequences and masks by repeating the first token."""
        if src.size(-2) % 2 == 0:
            pad_seqs = src
            pad_mask = mask
        else:
            pad_seqs = pt.cat([src[..., :1, :], src], dim=-2)
            if mask is None:
                pad_mask = mask
            else:
                # Prepend the first row, ...
                pad_mask = pt.cat([mask[..., :1, :], mask], dim=-2)
                # ... and the first column (incl. the just prepended row)
                pad_mask = pt.cat([pad_mask[..., :, :1], pad_mask], dim=-1)
        return pad_seqs, pad_mask

    def _shrink(
            self,
            pad_len: int,
            pad_mask: Tensor | None,
            is_causal: bool
    ) -> Tensors3T:
        """Generate cross-attention masks for compression and inflation."""
        if pad_mask is not None:
            inp = pt.maximum(pad_mask[..., 0::2, :], pad_mask[..., 1::2, :])
            shrunk = pt.maximum(inp[..., :, 0::2], inp[..., :, 1::2])
            out = pt.maximum(pad_mask[..., :, 0::2], pad_mask[..., :, 1::2])
        elif is_causal:
            i = pt.arange(pad_len).to(self.device)
            j = i[:pad_len // 2]
            inp = (i[None, :] <= (2 * j[:, None] + 1)).float().log()
            shrunk = None
            out = (2 * j[None, :] <= i[:, None]).float().log()
        else:
            inp = None
            shrunk = None
            out = None
        return inp, shrunk, out

    def forward(
            self,
            src: Tensor,
            attn_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
            is_causal: bool = True
    ) -> Tensor:
        """Forward pass through the compressor with optional masking.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of embeddings. Expected dimensions are
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
        # Apply positional encoding and dropout
        residual = self.drop(self.pos_enc(src))

        # Self-attend the original sequence
        mask = self.merge_masks(attn_mask, src_mask, is_causal)
        if self.norm_first:
            normed = self.norm_self_attn_inp(residual)
            attended = self.attend_inp(normed, mask, is_causal)
            residual = residual + self.drop(attended)
        else:
            attended = self.attend_inp(residual, mask, is_causal)
            residual = self.norm_self_attn_inp(residual + self.drop(attended))

        # Pad odd-length sequences and generate attention masks
        offset = residual.size(-2) % 2
        padded, padded_mask = self._pad(residual, mask)
        compress_mask, shrunk_mask, inflate_mask = self._shrink(
            padded.size(-2),
            padded_mask,
            is_causal
        )

        # Compress the sequence
        if self.norm_first:
            normalized_padded = self.norm_cross_attn_inp(padded)
            compressed, _ = self.compress(
                normalized_padded[:, 1::2, :],
                normalized_padded,
                normalized_padded,
                attn_mask=compress_mask,
                is_causal=False
            )
            residual = padded[:, 1::2, :] + self.drop(compressed)
        else:
            compressed, _ = self.compress(
                padded[:, 1::2, :],
                padded,
                padded,
                attn_mask=compress_mask,
                is_causal=False
            )
            residual = padded[:, 1::2, :] + self.drop(compressed)
            residual = self.norm_cross_attn_inp(residual)

        # Feed forward on the compressed sequence
        if self.norm_first:
            forwarded = self.forward_inp(self.norm_fwd_inp(residual))
            residual = residual + self.drop(forwarded)
        else:
            forwarded = self.forward_inp(residual)
            residual = self.norm_fwd_inp(residual + self.drop(forwarded))

        # Feed compressed sequence through wrapped model
        residual = self.model(residual, shrunk_mask, is_causal)

        # Inflate the sequence
        if self.norm_first:
            normed = self.norm_cross_attn_out(residual)
            inflated, _ = self.inflate(
                normalized_padded,
                normed,
                normed,
                attn_mask=inflate_mask,
                is_causal=False
            )
            residual = padded + self.drop(inflated)
        else:
            inflated, _ = self.inflate(
                padded,
                residual,
                residual,
                attn_mask=inflate_mask,
                is_causal=False
            )
            residual = self.norm_cross_attn_out(padded + self.drop(inflated))

        # Reduce odd-length sequences to their original "unpadded" length
        residual = residual[..., offset:, :]

        # Self-attend over the inflated sequence
        if self.norm_first:
            normed = self.norm_self_attn_out(residual)
            attended = self.attend_out(normed, mask, is_causal)
            residual = residual + self.drop(attended)
        else:
            attended = self.attend_out(residual, mask, is_causal)
            residual = self.norm_self_attn_out(residual + self.drop(attended))

        # Feed forward the inflated sequence
        if self.norm_first:
            forwarded = self.forward_out(self.norm_fwd_out(residual))
            residual = residual + self.drop(forwarded)
        else:
            forwarded = self.forward_out(residual)
            residual = self.norm_fwd_out(residual + self.drop(forwarded))

        return residual

    def reset_parameters(self) -> None:
        """Reset all learnable parameters in all components of the model."""
        self.model.reset_parameters()
        self.attend_inp.reset_parameters()
        self.attend_out.reset_parameters()
        self.forward_inp.reset_parameters()
        self.forward_out.reset_parameters()
        self.pos_enc.reset_parameters()
        self.compress._reset_parameters()
        self.norm_self_attn_inp.reset_parameters()
        self.norm_cross_attn_inp.reset_parameters()
        self.norm_fwd_inp.reset_parameters()
        self.inflate._reset_parameters()
        self.norm_self_attn_out.reset_parameters()
        self.norm_cross_attn_out.reset_parameters()
        self.norm_fwd_out.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Compressor
            A fresh, new instance of itself.

        """
        we_have_pos_enc = not isinstance(self.pos_enc, IdentityBlock)
        return self.__class__(
            self.model,
            self.attend_inp,
            self.forward_inp,
            self.pos_enc.new() if we_have_pos_enc else None,
            self.bias,
            self.dropout,
            self.norm_first,
            self.norm_cls,
            *self.args,
            device=self.device,
            dtype=self.dtype,
            **self.kwargs
        )
