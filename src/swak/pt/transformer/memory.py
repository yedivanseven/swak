from typing import Self, Any
import torch as pt
import torch.nn as ptn
from torch.nn import LayerNorm, RMSNorm
from ..types import Block, Tensor


# ToDo: Make Hierarchical Training loop
# ToDo: Add Memorizer
# ToDo: Add docstrings and unit tests
class MemorizerLayer(Block):

    def __init__(
            self,
            mod_dim: int,
            n_heads: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            batch_size: int = 16,
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
        compress = pt.empty(
            1,
            mem_size,
            mod_dim,
            device=device,
            dtype=dtype
        )
        ptn.init.xavier_uniform_(compress)
        self.compress = ptn.Parameter(compress)
        self.extract_update = ptn.MultiheadAttention(
            embed_dim=mod_dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype
        )
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
        self.norm_update = norm_cls(
            mod_dim,
            *args,
            device=device,
            dtype=dtype,
            **kwargs
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
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        return self.extract_update.in_proj_weight.device

    @property
    def dtype(self) -> pt.dtype:
        return self.extract_update.in_proj_weight.dtype

    def _update(self, src: Tensor, src_mask: Tensor | None) -> Tensor:
        normed_last_seqs = self.norm_update(self.last_seqs)
        memory_update, _ = self.extract_update(
            self.compress.expand(self.batch_size, -1, -1),
            normed_last_seqs,
            normed_last_seqs,
            key_padding_mask=self.last_mask,
            need_weights=False
        )
        updated_memory, _ = self.update_memory(
            self.memory,
            memory_update,
            memory_update,
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
        # Update at first inference call, then generate answer without
        memory = self._update(src, src_mask) if update else self.memory
        if self.norm_first:
            normed = self.norm_src(src)
            attended, _ = self.attend_to_memory(
                normed,
                memory,
                memory,
                need_weights=False
            )
            return src + self.drop(attended)
        else:
            attended, _ = self.attend_to_memory(
                src,
                memory,
                memory,
                need_weights=False
            )
            return self.norm_src(src + self.drop(attended))

    def forget(self, batch_size: int | None = None) -> None:
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
        ptn.init.xavier_uniform_(self.compress)
        self.extract_update._reset_parameters()
        self.update_memory._reset_parameters()
        self.attend_to_memory._reset_parameters()
        self.norm_update.reset_parameters()
        self.norm_src.reset_parameters()
        self.norm_memory.reset_parameters()
