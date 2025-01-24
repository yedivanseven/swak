from typing import Self
import torch as pt
from ..types import Tensor, Resettable
from ..exceptions import EmbeddingError
from .numerical import NumericalEmbedder
from .categorical import CategoricalEmbedder


class FeatureEmbedder(Resettable):
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
        if embed_num.mod_dim != embed_cat.mod_dim:
            msg = (f'Numerical ({embed_num.mod_dim}) and categorical ('
                   f'{embed_cat.mod_dim}) embedding dimensions must match!')
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
            Input tensor of must be of dtype ``float``. The last dimension is
            expected to contain first the values of all numerical features,
            followed by those of the categorical features.

        Returns
        -------
        Tensor
            The output tensor has one more dimension of size `mod_dim` added
            after the last dimension (with a size equal to the total number of
            features) than the `inp`, containing the stacked embeddings, first
            those of the numerical and then those of the categorical features.

        """
        return pt.cat([
            self.embed_num(inp[..., :self.n_num]),
            self.embed_cat(inp[..., self.n_num:].long())
        ], dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.embed_num.reset_parameters()
        self.embed_cat.reset_parameters()

    def new(
            self,
            embed_num: NumericalEmbedder | None = None,
            embed_cat: CategoricalEmbedder | None = None
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

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
