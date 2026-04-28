from typing import Self, Any
import torch as pt
import torch.nn as ptn
from torch.nn import LayerNorm, RMSNorm
from ..types import Block, Tensor


# ToDo: Add unit tests
class MemoryLayer(Block):
    """Encoder sub-layer that maintains a fixed-size, learnable memory.

    Incoming token sequences are enriched by cross-attending to a compressed
    summary of all previously seen chunks. Two attention operations govern
    the memory lifecycle: (1) extracting an update signal from the previous
    chunk and folding that signal into the current memory, and (2) letting the
    current input read from the updated memory.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Inputs are expected to be of that size in their
        last dimension.
    n_heads: int, optional
        Number of attention heads in each of the two attention operations.
        Defaults to 1.
    bias: bool, optional
        Whether to use bias in the attention projections.
        Defaults to ``False``.
    dropout: float, optional
        Dropout probability applied after the memory read. Defaults to 0.
    batch_size: int, optional
        Number of sequences processed in parallel. Defaults to 1.
    mem_size: int, optional
        Number of memory slots, i.e., the fixed size of the compressed memory
        along the sequence dimension. Defaults to 64.
    norm_first: bool, optional
        Whether to normalize inputs to the memory attention or the sum of
        inputs and outputs. Defaults to ``True``.
    norm_cls: type, optional
        Which type of norm to use between (sub-)layers. Must be one of
        ``torch.nn.LayerNorm`` (the default) or ``torch.nn.RMSNorm``.
    *args
        Additional arguments forwarded to each instantiation of `norm_cls`.
    device: str or torch.device, optional
        Torch device to create the layer on. Defaults to ``"cpu"``.
    dtype: torch.dtype, optional
        Torch dtype to create the layer in. Defaults to ``torch.float``.
    **kwargs
        Keyword arguments used to initialize an instance of `norm_cls`.

    See Also
    --------
    Memorizer

    """

    def __init__(
            self,
            mod_dim: int,
            n_heads: int = 1,
            bias: bool = False,
            dropout: float = 0.0,
            batch_size: int = 1,
            mem_size: int = 64,
            norm_first: bool = True,
            norm_cls: type[LayerNorm | RMSNorm] = LayerNorm,
            *args: Any,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.bias = bias
        self.dropout = dropout
        self.mem_size = mem_size
        self.norm_first = norm_first
        self.norm_cls = norm_cls
        self.args = args
        self.kwargs = kwargs
        self.last_seqs = pt.zeros(
            batch_size,
            1,
            mod_dim,
            device=device,
            dtype=dtype
        )
        self.last_seqs[:, :, 1::2] = 1.0
        self.last_mask = None
        memory = pt.empty(
            batch_size,
            mem_size,
            mod_dim,
            device=device,
            dtype=dtype
        )
        ptn.init.xavier_uniform_(memory)
        self.register_buffer('memory', memory)
        memory_mask = pt.zeros(1, mem_size, device=device, dtype=dtype)
        self.register_buffer('memory_mask', memory_mask)
        self.update_memory = ptn.MultiheadAttention(
            embed_dim=mod_dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype
        )
        self.attend_to_memory = ptn.MultiheadAttention(
            embed_dim=mod_dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype
        )
        self.norm_src = norm_cls(
            mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.norm_memory = norm_cls(
            mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.drop = ptn.Dropout(dropout)

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.update_memory.in_proj_weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.update_memory.in_proj_weight.dtype

    def _update(self, src: Tensor, src_mask: Tensor | None) -> Tensor:
        """Update memory with previous chunk and store current one for next."""
        combined_memory = pt.concat([self.memory, self.last_seqs], dim=-2)
        if self.last_mask is None:
            combined_mask = None
        else:
            combined_mask = pt.concat([
                self.memory_mask.expand(self.batch_size, -1),
                self.last_mask,
            ], dim=-1)
        updated_memory, _ = self.update_memory(
            self.memory,
            combined_memory,
            combined_memory,
            key_padding_mask=combined_mask,
            need_weights=False
        )
        normed_memory = self.norm_memory(updated_memory)
        self.memory.copy_(normed_memory.detach())
        self.last_seqs = src.clone()
        self.last_mask = src_mask
        return normed_memory

    def forward(
            self,
            src: Tensor,
            src_mask: Tensor | None = None,
            update: bool = True
    ) -> Tensor:
        """Forward pass through the memory layer.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of token embeddings with dimensions
            (`B`, `S`, `D`), with batch size `B`, sequence length `S`,
            and model dimension `D`.
        src_mask: Tensor, optional
            Floating-point padding mask with a shape broadcastable to
            (`B`, `S`). A value of 0.0 indicates that a token *should* be
            attended to and ``"-inf"`` that it should *not*. Stored alongside
            `src` for use in the next :meth:`forward` call.
            Defaults to ``None``.
        update: bool, optional
            Whether to update the memory before reading from it. Set to
            ``True`` (the default) during training and on the first call of
            each new user turn during inference. Set to ``False`` during
            autoregressive token generation to freeze memory.

        Returns
        -------
        Tensor
            Input enriched with information read from memory, with the same
            shape as `src`.

        """
        if self.norm_first:
            normed = self.norm_src(src)
            memory = self._update(normed, src_mask) if update else self.memory
            attended, _ = self.attend_to_memory(
                normed,
                memory,
                memory,
                need_weights=False
            )
            return src + self.drop(attended)
        else:
            memory = self._update(src, src_mask) if update else self.memory
            attended, _ = self.attend_to_memory(
                src,
                memory,
                memory,
                need_weights=False
            )
            return self.norm_src(src + self.drop(attended))

    def forget(self, batch_size: int | None = None) -> None:
        """Reset memory between independent sequences/documents/conversations.

        Parameters
        ----------
        batch_size: int, optional
            The batch size for the next sequences. If ``None``, the current
            batch size is retained. Defaults to ``None``.

        """

        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.last_seqs = pt.zeros(
            self.batch_size,
            1,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.last_seqs[:, :, 1::2] = 1.0
        self.last_mask = None
        memory = pt.empty(
            self.batch_size,
            self.mem_size,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        ptn.init.xavier_uniform_(memory)
        self.register_buffer('memory', memory)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same hyperparameters.

        Returns
        -------
        MemoryLayer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.batch_size,
            self.mem_size,
            self.norm_first,
            self.norm_cls,
            *self.args,
            device=self.device,
            dtype=self.dtype,
            **self.kwargs
        )

    def reset_parameters(self) -> None:
        """Reset all learnable parameters and the memory of the layer."""
        self.last_seqs = pt.zeros(
            self.batch_size,
            1,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        self.last_seqs[:, :, 1::2] = 1.0
        self.last_mask = None
        ptn.init.xavier_uniform_(self.memory)
        self.update_memory._reset_parameters()
        self.attend_to_memory._reset_parameters()
        self.norm_src.reset_parameters()
        self.norm_memory.reset_parameters()
