from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Mixer, Tensor


class CrossAttentionMixer(Mixer):
    """Combine stacked feature vectors via cross-attention with learned query.

    Similar to, but lighter than self-attention and arguably cleaner as
    a mixer, since the mixing intent is decoupled from the input content.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    n_heads: int, optional
        Number of attention heads. Must evenly divide `mod_dim`.
        Defaults to 1.
    bias: bool, optional
        Whether to add learnable bias vectors in the attention projections.
        Defaults to ``True``.
    dropout: float, optional
        The amount of dropout to apply to the attention weights as well as
        to the mixed-features output. Defaults to 0.
    skip: bool, optional
        Whether to add a residual connection around the feature mixing.
        Defaults to ``True``.
    keep_dim: bool, optional
       Whether to keep the next-to-last dimension of the output tensor as 1
       or squeeze it. Defaults to ``False``.
    device: str or torch.device, optional
        Torch device to first create the mixer on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the mixer in.
        Defaults to ``torch.float``.

    See Also
    --------
    SelfAttentionMixer

    """

    def __init__(
            self,
            mod_dim: int,
            n_heads: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            skip: bool = True,
            keep_dim: bool = False,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.n_heads = n_heads
        self.bias = bias
        self.dropout = dropout
        self.skip = skip
        self.keep_dim = keep_dim
        self.drop = ptn.Dropout(dropout)
        self.query = ptn.Parameter(
            pt.empty(1, mod_dim, device=device, dtype=dtype)
        )
        ptn.init.xavier_uniform_(self.query)
        self.attention = ptn.MultiheadAttention(
            embed_dim=mod_dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.attention.out_proj.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.attention.out_proj.weight.dtype

    def importance(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Per-feature attention weights derived from the learned query.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None, optional
            Padding mask with its last dimension of size `n_features`. For a
            binary mask, ``True`` values indicates that the corresponding
            feature will be ignored. For a float mask, the value will be
            directly added to the corresponding attention-key value.
            Defaults to ``None``.

        Returns
        -------
        Tensor
            Attention weights derived from the learned query. The output
            tensor has one fewer dimensions than the `inp` with the last
            dimension being dropped.

        """
        query = self.query.expand(*inp.shape[:-2], -1, -1)
        _, scores = self.attention(
            query, inp, inp,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True
        )
        return scores.squeeze(dim=-2)

    def forward(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None, optional
            Padding mask with its last dimension of size `n_features`. For a
            binary mask, ``True`` values indicates that the corresponding
            feature will be ignored. For a float mask, the value will be
            directly added to the corresponding attention-key value.
            Defaults to ``None``.

        Returns
        -------
        Tensor
            Depending on `keep_dim`, the output tensor has the same number of
            dimensions as `inp` or one fewer. The next-to-last dimension is
            either 1 or dropped. The last dimension (of size `mod_dim`)
            contains the cross-attention-pooled combination of all feature
            vectors.

        """
        if inp.size(-2) == 0:
            return inp if self.keep_dim else inp.sum(dim=-2)
        query = self.query.expand(*inp.shape[:-2], -1, -1)
        mixed, _ = self.attention(
            query, inp, inp,
            key_padding_mask=mask,
            need_weights=False
        )
        out = self.drop(mixed)
        if self.skip:
            out += inp.mean(dim=-2, keepdim=True)
        return out if self.keep_dim else out.squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        ptn.init.xavier_uniform_(self.query)
        self.attention._reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        CrossAttentionMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.skip,
            self.keep_dim,
            self.device,
            self.dtype
        )


class SelfAttentionMixer(Mixer):
    """Combine stacked feature vectors via multi-head self-attention pooling.

    Each feature vector attends to all others, and the resulting attended
    representations are averaged across the feature dimension to yield a single
    output vector per instance. The attention weights averaged over all query
    positions serve as per-instance feature importance scores.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    n_heads: int, optional
        Number of attention heads. Must evenly divide `mod_dim`.
        Defaults to 1.
    bias: bool, optional
        Whether to add learnable bias vectors in the attention projections.
        Defaults to ``True``.
    dropout: float, optional
        The amount of dropout to apply to the attention weights as well as
        to the mixed-features output. Defaults to 0.
    skip: bool, optional
        Whether to add a residual connection around the feature mixing.
        Defaults to ``True``.
    keep_dim: bool, optional
        Whether to keep the next-to-last dimension of the output tensor as 1
        or squeeze it. Defaults to ``False``.
    device: str or torch.device, optional
        Torch device to first create the embedder on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the embedder in.
        Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            n_heads: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            skip: bool = True,
            keep_dim: bool = False,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.n_heads = n_heads
        self.bias = bias
        self.dropout = dropout
        self.skip = skip
        self.keep_dim = keep_dim
        self.drop = ptn.Dropout(dropout)
        self.attention = ptn.MultiheadAttention(
            embed_dim=mod_dim,
            num_heads=n_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.attention.out_proj.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.attention.out_proj.weight.dtype

    def importance(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Per-instance weights in the attention-based combination of features.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None, optional
            Padding mask with its last dimension of size `n_features`. For a
            binary mask, ``True`` values indicates that the corresponding
            feature will be ignored. For a float mask, the value will be
            directly added to the corresponding attention-key value.
            Defaults to ``None``.

        Returns
        -------
        Tensor
            Attention weights averaged over query positions. The output tensor
            has one fewer dimensions than the `inp` with the last dimension
            being dropped.

        """
        _, scores = self.attention(
            inp, inp, inp,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True
        )
        return scores.mean(dim=-2)

    def forward(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None, optional
            Padding mask with its last dimension of size `n_features`. For a
            binary mask, ``True`` values indicates that the corresponding
            feature will be ignored. For a float mask, the value will be
            directly added to the corresponding attention-key value.
            Defaults to ``None``.

        Returns
        -------
        Tensor
            Depending on `keep_dim`, the output tensor has the same number of
            dimensions as `inp` or one fewer. The next-to-last dimension is
            either 1 or dropped. The last dimension (of size `mod_dim`)
            contains the attention-pooled combination of all feature vectors.

        """
        if inp.size(-2) == 0:
            return inp if self.keep_dim else inp.sum(dim=-2)
        mixed, _ = self.attention(
            inp, inp, inp,
            key_padding_mask=mask,
            need_weights=False
        )
        out = self.drop(mixed.mean(dim=-2, keepdim=True))
        if self.skip:
            out += inp.mean(dim=-2, keepdim=True)
        return out if self.keep_dim else out.squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.attention._reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        SelfAttentionMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.skip,
            self.keep_dim,
            self.device,
            self.dtype
        )
