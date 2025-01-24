from typing import Any, Self
from functools import cached_property
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Resettable
from .activated import ActivatedEmbedder
from .gated import GatedEmbedder
from .gated_residual import GatedResidualEmbedder

type EmbCls = type[ActivatedEmbedder | GatedEmbedder | GatedResidualEmbedder]


class NumericalEmbedder(Resettable):
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
    **args
        Additional arguments to use when instantiating `emb_cls`.
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
            emb_cls: EmbCls,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.emb_cls = emb_cls
        self.args = args
        self.kwargs = kwargs
        self.embed = ptn.ModuleList(
            [emb_cls(mod_dim, *args, **kwargs) for _ in self.features]
        )

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
        return pt.stack(emb or self.mod_dim * [pt.zeros(*inp.shape)], self.dim)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        for emb in self.embed:
            emb.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            n_features: int | None = None,
            emb_cls: EmbCls | None = None,
            *args: Any,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Desired embedding size. Will become the size of the last dimension
            the output tensor. Overwrites the `mod_dim` of the current
            instance if given. Defaults to ``None``.
        n_features: int, optional
            Number of features to embed, which must equal the size of the last
            dimension of the input tensor. Overwrites the `n_features` of the
            current instance if given. Defaults to ``None``.
        emb_cls: type, optional
            The PyTorch module to use as embedding class. Must take `mod_dim`
            as its first argument on instantiation. Overwrites the `emb_cls`
            of the current instance if given. Defaults to ``None``.
        *args
            Additional arguments replace those of the current instance and are
            then used when instantiating `emb_cls`.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then used together when
            instantiating `emb_cls`.

        Returns
        -------
        NumericalEmbedder
            A fresh, new instance of itself.

        See Also
        --------
        ActivatedEmbedder
        GatedEmbedder

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.n_features if n_features is None else n_features,
            self.emb_cls if emb_cls is None else emb_cls,
            *args,
            **(self.kwargs | kwargs)
        )
