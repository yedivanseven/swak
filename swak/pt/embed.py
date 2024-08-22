from typing import Any, Self
from collections.abc import Callable, Iterable
from functools import cached_property
import torch as pt
import torch.nn as ptn
from .types import Tensor, Module
from .exceptions import EmbeddingError

type Gate = Callable[[Tensor], Tensor]

__all__ = [
    'GluEmbedder',
    'NumericalEmbedder',
    'CategoricalEmbedder',
    'FeatureEmbedder'
]


class GluEmbedder(Module):
    """Flexible Gated Linear Unit (GLU) embedding for numerical features.

    Parameters
    ----------
    out_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    gate: Module, optional
        The activation function to be applied to half of the (linearly)
        transformed input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``.
        Defaults to a sigmoid.
    inp_dim: int, optional
        The number of features to embed. Defaults to 1.
    **kwargs
        Additional keyword arguments to pass through to the linear layer.

    """

    def __init__(
            self,
            out_dim: int,
            gate: Module | Gate = ptn.Sigmoid(),
            inp_dim: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.gate = gate
        self.inp_dim = inp_dim
        self.kwargs = kwargs
        self.embed = ptn.Linear(inp_dim, 2 * out_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Embed a single numerical feature through a Gated Linear Unit (GLU).

        Parameters
        ----------
        inp: Tensor
            Input tensor with at least 2 dimensions. The last dimension is
            typically expected to be of size 1 and to contain the numerical
            value of a single feature. In case `inp_dim` dim was explicitly
            set to a value > 1 on instantiation, the size of the last
            dimension must match `inp_dim`, the number of numerical features
            to embed together.

        Returns
        -------
        Tensor
            The output has the same number of dimensions as the input with the
            size of the last dimension changed to the specified `out_dim`.

        """
        emb = self.embed(inp)
        return emb[..., :self.out_dim] * self.gate(emb[..., self.out_dim:])

    def new(
            self,
            out_dim: int | None = None,
            gate: Module | Gate | None = None,
            inp_dim: int | None = None,
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
        gate: Module, optional
            The activation function to be applied to half of the (linearly)
            transformed input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `out_dim` of the current instance if given.
            Defaults to ``None``.
        inp_dim: int, optional
            The number of features to embed. Overwrites the `inp_dim` of the
            current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        GluEmbedder
            A fresh, new instance of itself.

        Notes
        -----

        """
        return self.__class__(
            self.out_dim if out_dim is None else out_dim,
            self.gate if gate is None else gate,
            self.inp_dim if inp_dim is None else inp_dim,
            **(self.kwargs | kwargs)
        )


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


class FeatureEmbedder(Module):
    """Jointly embed numerical and categorical features into stacked vectors.

    Given a float tensor where both, numerical and categorical features
    appear (one before the other in the last dimension), instances of this
    class treat them on equal footing and produce stacked embedding vectors
    for all of them.

    Parameters
    ----------
    embed_num: NumericalEmbedder
        A fully configured instance of ``NumericalEmbedder``.
    embed_cat: CategoricalEmbedder
        A fully configured instance of ``CategoricalEmbedder``.

    Raises
    ------
    EmbeddingError
        If the embedding dimension of the numerical and the categorical
        embedders do not match.

    """

    def __init__(
            self,
            embed_num: NumericalEmbedder,
            embed_cat: CategoricalEmbedder
    ) -> None:
        super().__init__()
        if embed_num.out_dim != embed_cat.out_dim:
            msg = (f'Numerical ({embed_num.out_dim}) and categorical ('
                   f'{embed_cat.out_dim}) embedding dimensions must match!')
            raise EmbeddingError(msg)
        self.embed_num = embed_num
        self.embed_cat = embed_cat

    @property
    def n_num(self) -> int:
        """Number of numerical features."""
        return self.embed_num.n_features

    @property
    def n_cat(self) -> int:
        """Number of categorical features."""
        return self.embed_cat.n_features

    @property
    def n_features(self) -> int:
        """Total number of features."""
        return self.n_num + self.n_cat

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for numerical and categorical feature embeddings.

        Parameters
        ----------
        inp: Tensor
            Input tensor of dtype float with at least 2 dimensions. The last
            dimension is expected to contain first the values of all numerical
            features, followed by those of the categorical features.

        Returns
        -------
        Tensor
            The output tensor has one more dimension of size `out_dim` added
            after the last dimension (with a size equal to the total number of
            features`) than the `inp`, containing the stacked embeddings, first
            those fo the numerical and then those of teh categorical features.

        """
        embedded = [
            self.embed_num(inp[..., :self.n_num]),
            self.embed_cat(inp[..., self.n_num:].long())
        ]
        return pt.cat(embedded, dim=-2)

    def new(
            self,
            embed_num: NumericalEmbedder | None = None,
            embed_cat: CategoricalEmbedder | None = None
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Needed to reset all parameters when the class or its initialization
        parameters are not readily at hand at the point in the code where a
        reset is desired.

        Parameters
        ----------
        embed_num: NumericalEmbedder, optional
            Overwrites the `embed_num` of the current instance if given.
            Defaults to ``None``.
        embed_cat: CategoricalEmbedder, optional
            Overwrites the `embed_cat` of the current instance if given.
            Defaults to ``None``.

        Returns
        -------
        FeatureEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.embed_num.new() if embed_num is None else embed_num,
            self.embed_cat.new() if embed_cat is None else embed_cat
        )
