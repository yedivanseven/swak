from typing import Any, Self
from collections.abc import Iterable
from functools import cached_property
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module


class CategoricalEmbedder(Module):
    """Embed one or more categorical features as numerical vectors.

    Parameters
    ----------
    out_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    cat_count: int or iterable of int, optional
        One integer or an iterable (e.g., a tuple or list) of integers, each
        specifying the total number of categories in the respective feature.
        Defaults to an emtpy tuple.
    *cat_counts: int
        Category counts for additional features. Together with the `cat_count`,
        the total number of category counts, i.e., the total number of features
        to embed must match the size of the last dimension of the input tensor.
    **kwargs
        Keyword arguments are forwarded to the PyTorch ``Embedding`` class.

    Notes
    -----
    The integer numbers identifying a category are expected to be zero-base,
    i.e., if the category count of a feature is 3, the allowed category
    identifier are 0, 1, and 2. If you need a padding index (e.g., to mark
    missing/unknown values), do not forget to increase all `cat_counts` by one!

    """

    def __init__(
            self,
            out_dim: int,
            cat_count: int | Iterable[int] = (),
            *cat_counts: int,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        cat_count = self.__valid(cat_count)
        self.cat_counts: tuple[int, ...] = cat_count + self.__valid(cat_counts)
        self.kwargs = kwargs
        self.embed = ptn.ModuleList([
            ptn.Embedding(count, out_dim, **kwargs)
            for count in self.cat_counts
        ])

    @property
    def n_features(self) -> int:
        """Number of features to embed."""
        return len(self.cat_counts)

    @cached_property
    def features(self) -> range:
        """Range of feature indices."""
        return range(self.n_features)

    @cached_property
    def dim(self) -> int:
        """The output tensor dimension index to stack features into."""
        return -1 if self.n_features < 1 else -2

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for embedding categorical features into vectors.

        Parameters
        ----------
        inp: tensor
            Input tensor of type ``long`` with at least 2 dimensions. The size
            of the last dimension is expected to match the number of specified
            `cat_counts` and to contain the integer identifiers of the
            categories in the respective feature. These identifiers must all
            be lower in value than their respective count.

        Returns
        -------
        Tensor
            The output tensor has one more dimension of size `out_dim` added
            after the last dimension (with a size equal to the number of
            `cat_counts`) than the `inp`, containing the stacked embeddings.

        """
        emb = [self.embed[cat](inp[..., cat]) for cat in self.features]
        return pt.stack(emb or self.out_dim * [pt.zeros(*inp.shape)], self.dim)

    def new(
            self,
            out_dim: int | None = None,
            cat_count: int | Iterable[int] | None = None,
            *cat_counts: int,
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
            of the output tensor. Overwrites the `out_dim` of the current
            instance if given. Defaults to ``None``.
        cat_count: int or iterable of int, optional
            One integer or an iterable (e.g., tuple or list) of integers, each
            specifying the number of categories in the respective feature.
            Overwrites the `cat_counts` of the current instance if given.
            Defaults to ``None``.
        *cat_counts: int
            Category counts for additional features. Together with the
            `cat_count`, the total number of category counts must match the
            size of the last dimension of the input tensor.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then used together for
            instantiating the PyTorch ``Embedding`` class.

        Returns
        -------
        CategoricalEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.out_dim if out_dim is None else out_dim,
            self.cat_counts if cat_count is None else cat_count,
            *cat_counts,
            **(self.kwargs | kwargs)
        )

    @staticmethod
    def __valid(cat_counts: int | Iterable[int]) -> tuple[int, ...]:
        """Ensure that the argument is indeed an iterable of integers."""
        try:
            return tuple(abs(int(cat_count)) for cat_count in cat_counts)
        except TypeError:
            return abs(int(cat_counts)),
