from typing import Any, Self
from functools import cached_property
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Bag, Block


class NumericalEmbedder(Bag):
    """Transform (scalar) numerical features into embedding vectors.

    Parameters
    ----------
    mod_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    n_features: int
        Number of features to embed, which must equal the size of the last
        dimension of the input tensor.
    emb_cls: type
        The PyTorch module to use as embedding class. Must take `mod_dim` as
        its first argument on instantiation, take tensors of size 1 in their
        last dimension, and change that dimension to size `mod_dim`.
    *args
        Additional arguments to use when instantiating `emb_cls`.
    device: str or torch.device, optional
        Torch device to first create the embedder on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the embedder in.
        Defaults to ``torch.float``.
    **kwargs
        Additional keyword arguments to use when instantiating `emb_cls`.

    See Also
    --------
    ActivatedEmbedder
    GatedEmbedder
    GatedResidualEmbedder

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            emb_cls: type[Block],
            *args: Any,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.__n_features = n_features
        self.emb_cls = emb_cls
        self.args = args
        self.kwargs = kwargs
        self.embed = ptn.ModuleList(
            [emb_cls(mod_dim, *args, **kwargs) for _ in self.features]
        ).to(device=device, dtype=dtype)

    def __bool__(self) -> bool:
        return bool(self.embed)

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def n_features(self) -> int:
        """The number of features to embed."""
        return self.__n_features

    @property
    def device(self) -> pt.device | None:
        """The device the embedders live on, or None if there aren't any."""
        if not self:
            return None
        return getattr(self.embed[0], 'weight', self.embed[0]).device

    @property
    def dtype(self) -> pt.dtype | None:
        """The dtype of the embedders, or None if there aren't any."""
        if not self:
            return None
        return getattr(self.embed[0], 'weight', self.embed[0]).dtype

    @cached_property
    def features(self) -> range:
        """Range of feature indices."""
        return range(self.n_features)

    @cached_property
    def dim(self) -> int:
        """The output tensor dimension index to stack features into."""
        return -1 if self.n_features < 1 else -2

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for embedding scalar numerical features into vectors.

        Parameters
        ----------
        inp: Tensor
            The last dimension of the input tensor is expected to be of size
            `n_features` and to contain the scalar values of the individual
            numerical features.

        Returns
        -------
        Tensor
            The output tensor has one more dimension of size `mod_dim` added
            after the last dimension (of size `n_features`) than the `inp`,
            containing the stacked embeddings.

        """
        emb = [self.embed[f](inp[..., [f]]) for f in self.features]
        return pt.stack(emb or self.mod_dim * [
            pt.zeros(*inp.shape, device=inp.device, dtype=inp.dtype)
        ], self.dim)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        for emb in self.embed:
            emb.reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        NumericalEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_features,
            self.emb_cls,
            *self.args,
            device=self.device,
            dtype=self.dtype,
            **self.kwargs
        )
