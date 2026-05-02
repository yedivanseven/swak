import warnings
from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, PosEnc, Trafo
from ..blocks import IdentityBlock
from .layer import EncoderLayer
from .memory import MemoryLayer


# ToDo: Make Hierarchical Training loop
# ToDo. Add unit tests
class Memorizer(Trafo):
    """Transformer encoder stack with built-in, learnable memory.

    During training, an outer loop must provides a batch of long sequences
    and an inner loop must feed chunks of these sequences to this ``Module``
    with ``update=True``. Before each fresh batch, the :meth:`forget` method
    must be called to reset the memory. During inference, use ``update=True``
    on the first call, then autoregressively generate tokens one at a time
    with ``update=False``. Return the model answer to the user, and pass
    that answer together the user's new request again with ``update=True``.

    Parameters
    ----------
    memory: MemoryLayer
        A suitably parameterized instance of :class:`MemoryLayer`
    layer: EncoderLayer
        A suitably parameterized instance of :class:`EncoderLayer`.
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
        Apply dropout to the sum of token embeddings and positional encodings
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

    Note
    ----
    If the `layer` sets `norm_first` to ``True``, no norm is applied to the
    final output of the :class:`Encoder`. If a trailing norm is desired,
    it should be applied externally, after this module.

    See Also
    --------
    MemoryLayer
    EncoderLayer
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            memory: MemoryLayer,
            layer: EncoderLayer,
            n_layers: int = 1,
            pos_enc: PosEnc | None = None,
            dropout: float = 0.0,
            device: pt.device = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.n_layers = self.__valid(n_layers)
        self.memories = ptn.ModuleList([
            memory.new().to(device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ])
        self.layers = ptn.ModuleList([
            layer.new().to(device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ])
        pos_enc = pos_enc or IdentityBlock(layer.mod_dim)
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.dropout = dropout
        self.drop = ptn.Dropout(dropout)
        self.__offset = 1

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

    @property
    def has_pos_enc(self) -> bool:
        """Whether positional encodings are applied."""
        return (
            self.layers[0].has_pos_enc or
            not isinstance(self.pos_enc, IdentityBlock)
        )

    @property
    def offset(self) -> int:
        """The number of tokens processed so far."""
        return self.__offset

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

    def forward(
            self,
            src: Tensor,
            attn_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
            is_causal: bool = True,
            update: bool = True
    ) -> Tensor:
        """Forward pass through the remembering transformer encoder.

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
        update: bool, optional
            Whether to update the memory in the forward pass. During training,
            for passing generated answers and user requests, this should be
            ``True``, which is the default. Set to ``False``only when
            autoregressively gerenating tokens one by one..

        Returns
        -------
        Tensor
            Transformed input with again dimensions (..., `S`, `D`).

        Important
        ---------
        Boolean attention masks are not accepted!

        """
        mask = self.merge_masks(attn_mask, src_mask, is_causal)
        out = self.drop(self.pos_enc(src, self.offset))
        for layer in range(self.n_layers):
            out = self.memories[layer](out, src_mask, update)
            out = self.layers[layer](out, mask, is_causal)
        self.__offset += out.size(dim=-2)
        return out

    def forget(self, batch_size: int | None = None) -> None:
        """Forget and re-initialize the memory in all layers.

        Parameters
        ----------
        batch_size: int, optional
            The batch size for the next (long) sequences to process.

        """
        for memory in self.memories:
            memory.forget(batch_size)
        self.__offset = 1

    def reset_parameters(self) -> None:
        """Reset all learnable parameters in all components of the model."""
        for layer in range(self.n_layers):
            self.memories[layer].reset_parameters()
            self.layers[layer].reset_parameters()
        self.pos_enc.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Encoder
            A fresh, new instance of itself.

        """
        we_have_pos_enc = not isinstance(self.pos_enc, IdentityBlock)
        return self.__class__(
            self.memories[0],
            self.layers[0],
            self.n_layers,
            self.pos_enc.new() if we_have_pos_enc else None,
            self.dropout,
            self.device,
            self.dtype
        )
