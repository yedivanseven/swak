from typing import Self
import torch as pt
from ..types import Tensor, Bag
from ..exceptions import EmbeddingError
from .numerical import NumericalEmbedder
from .categorical import CategoricalEmbedder


class FeatureEmbedder(Bag):
    """Jointly embed numerical and categorical features into stacked vectors.

    Given a float tensor where both, numerical and categorical features
    appear (one before the other in the last dimension), instances of this
    class treat them on equal footing and produce stacked embedding vectors
    for all of them.

    Parameters
    ----------
    num: NumericalEmbedder
        A fully configured instance of :class:`NumericalEmbedder`.
    cat: CategoricalEmbedder
        A fully configured instance of :class:`CategoricalEmbedder`.
    device: str or pt.device, optional
        Torch device to first create the embedders on. Defaults to "cpu".
    dtype: pt.dtype, optional
        Torch dtype to first create the embedders in.
        Defaults to ``torch.float``.

    Raises
    ------
    EmbeddingError
        If the embedding dimension, devices, or dtypes of the numerical and
        the categorical embedders do not match (provided they are even set).

    See Also
    --------
    NumericalEmbedder
    CategoricalEmbedder

    """

    def __init__(self,
            num: NumericalEmbedder,
            cat: CategoricalEmbedder,
            device: pt.device = 'cpu',
            dtype: pt.dtype = pt.float,
    ) -> None:
        super().__init__()
        self.num = num.to(device=device, dtype=dtype)
        self.cat = cat.to(device=device, dtype=dtype)
        self.__raise_on_incompatible()

    def __raise_on_incompatible(self) -> None:
        """Raise if mod_dim, device, or dtype are different but not None."""
        for attribute in ('mod_dim', 'device', 'dtype'):
            num = getattr(self.num, attribute)
            cat = getattr(self.cat, attribute)
            if num is not None and cat is not None and num != cat:
                tmp = ('The "{}" of the numerical ({}) and categorical '
                       '({}) embedders does not match!')
                msg = tmp.format(attribute, num, cat)
                raise EmbeddingError(msg)

    @property
    def device(self) -> pt.device | None:
        """The device the embedders live on, or None if there aren't any."""
        return self.num.device or self.cat.device or None

    @property
    def dtype(self) -> pt.dtype | None:
        """The dtype of the embedders, or None if there aren't any."""
        return self.num.dtype or self.cat.dtype or None

    @property
    def mod_dim(self) -> int:
        """The dimension of the embedding vectors."""
        return self.num.mod_dim

    @property
    def n_num(self) -> int:
        """Number of numerical features."""
        return self.num.n_features

    @property
    def n_cat(self) -> int:
        """Number of categorical features."""
        return self.cat.n_features

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
            self.num(inp[..., :self.n_num]),
            self.cat(inp[..., self.n_num:].long())
        ], dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.num.reset_parameters()
        self.cat.reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        FeatureEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(self.num.new(), self.cat.new())
