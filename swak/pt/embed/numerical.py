from typing import Any, Self
from functools import cached_property
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module


class NumericalEmbedder(Module):
    """Transform (scalar) numerical features into embedding vectors.

    Parameters
    ----------
    out_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    n_features: int
        Number of features to embed, which must equal the size of the last
        dimension of the input tensor.
    emb_cls: type
        The PyTorch module to use as embedding class. Must take `out_dim` as
        its first argument on instantiation, take tensors of size 1 in their
        last dimension and append a dimension of size `out_dim` to them.
    **kwargs
        Additional keyword arguments to use when instantiating `emb_cls`.

    """

    def __init__(
            self,
            out_dim: int,
            n_features: int,
            emb_cls: type[Module],
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_features = n_features
        self.emb_cls = emb_cls
        self.kwargs = kwargs
        self.embed = ptn.ModuleList(
            [emb_cls(out_dim, **kwargs)] * n_features
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
            Input tensor with at least 2 dimensions. The last dimension is
            expected to be of size `n_features`. and to contain the scalar
            values of the individual  numerical features.

        Returns
        -------
        Tensor
            The output tensor has one more dimension of size `out_dim` added
            after the last dimension (of size `n_features`) than the `inp`,
            containing the stacked embeddings.

        """
        emb = [self.embed[f](inp[..., [f]]) for f in self.features]
        return pt.stack(emb or self.out_dim * [pt.zeros(*inp.shape)], self.dim)

    def new(
            self,
            out_dim: int | None = None,
            n_features: int | None = None,
            emb_cls: type[Module] | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Needed to reset all parameters when the class or its initialization
        parameters are not readily at hand at the point in the code where a
        reset is desired.

        Parameters
        ----------
        out_dim: int, optional
            Desired embedding size. Will become the size of the last dimension
            the output tensor. Overwrites the `out_dim` of the current
            instance if given. Defaults to ``None``.
        n_features: int, optional
            Number of features to embed, which must equal the size of the last
            dimension of the input tensor. Overwrites the `n_features` of the
            current instance if given. Defaults to ``None``.
        emb_cls: type, optional
            The PyTorch module to use as embedding class. Must take `out_dim`
            as its first argument on instantiation. Overwrites the `emb_cls`
            of the current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then used together when
            instantiating `emb_cls`.

        Returns
        -------
        NumericalEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.out_dim if out_dim is None else out_dim,
            self.n_features if n_features is None else n_features,
            self.emb_cls if emb_cls is None else emb_cls,
            **(self.kwargs | kwargs)
        )
