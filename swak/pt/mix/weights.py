from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Mixer, Tensor


class GlobalWeightsMixer(Mixer):
    """Combine stacked feature vectors through a learnable linear combination.

    A single, global set of linear-combination coefficients is learned and
    shared across all instances. The coefficients sum to 1 via softmax and
    can thus be seen as some sort of global feature importance.

    Parameters
    ----------
    mod_dim: int
        Ignored but mandatory to maintain API compatibility.
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.
    dropout: float, optional
        The amount of dropout to apply to the mixed-features output.
        Defaults to 0.
    skip: bool, optional
        Whether to add a residual connection around the feature mixing.
        Defaults to ``True``.
    keep_dim: bool, optional
        Whether to keep the next-to-last dimension of the output tensor as 1
        or squeeze it. Defaults to ``False``.
    device: str or pt.device, optional
        Torch device to first create the mixer on. Defaults to "cpu".
    dtype: pt.dtype, optional
        Torch dtype to first create the mixer in.
        Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            dropout: float = 0.0,
            skip: bool = True,
            keep_dim: bool = False,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.__n_features = n_features
        self.dropout = dropout
        self.skip = skip
        self.keep_dim = keep_dim
        self.drop = ptn.Dropout(dropout)
        self.coeffs = ptn.Parameter(
            pt.ones(n_features, device=device, dtype=dtype)
        )
        self.score = ptn.Softmax(dim=-1)

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def n_features(self) -> int:
        """The number of features in the bag."""
        return self.__n_features

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.coeffs.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.coeffs.dtype

    def importance(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Learned global feature weights in the normed sum over all features.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last dimension is expected to match the
            `n_features` provided at instantiation. The last dimension (of
            size `mod_dim`) is expected to contain the feature vectors.
        mask: Tensor or None
            Ignored for the :class:`GlobalWeightsMixer`.

        Returns
        -------
        Tensor
            The softmax-normalized coefficients broadcast to the shape of
            `inp` with the last dimension dropped, i.e. ``(..., n_features)``.

        """
        return self.score(self.coeffs.expand(*inp.shape[:-1]))

    def forward(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last dimension is expected to match the
            `n_features` provided at instantiation. The last dimension (of
            size `mod_dim`) is expected to contain the feature vectors.
        mask: Tensor or None
            Ignored for the :class:`GlobalWeightsMixer`.

        Returns
        -------
        Tensor
            Depending on `keep_dim`, the output tensor has the same number of
            dimensions as `inp` or one fewer. The next-to-last dimension is
            either 1 or dropped. The last dimension (of size `mod_dim`)
            contains the globally weighted linear combination of all feature
            vectors.

        """
        if self.n_features == 0:
            return inp if self.keep_dim else inp.sum(dim=-2)
        out = self.drop(self.importance(inp).unsqueeze(dim=-2) @ inp)
        if self.skip:
            out += inp.mean(dim=-2, keepdim=True)
        return out if self.keep_dim else out.squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize the coefficients for the linear combination."""
        ptn.init.constant_(self.coeffs, 1.0)

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        GlobalMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_features,
            self.dropout,
            self.skip,
            self.keep_dim,
            self.device,
            self.dtype
        )


class InstanceWeightsMixer(Mixer):
    """Combine stacked feature vectors by a per-instance linear combination.

    The per-instance coefficients sum to 1 for each data point and can thus be
    seen as some sort of per-instance feature importance. They are obtained by
    concatenating all features into a single, wide vector, linearly projecting
    down to a vector with the same number of elements as there are features to
    combine, and then applying a softmax.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.
    bias: bool, optional
        Whether to add a learnable bias vector in the projection.
        Defaults to ``True``.
    dropout: float, optional
        The amount of dropout to apply to the mixed-features output.
        defaults to 0.
    skip: bool, optional
        Whether to add a residual connection around the feature mixing.
        Defaults to ``True``.
    keep_dim: bool, optional
        Whether to keep the next-to-last dimension of the output tensor as 1
        or squeeze it. Defaults to ``False``.
    device: str or pt.device, optional
        Torch device to first create the embedder on. Defaults to "cpu".
    dtype: pt.dtype, optional
        Torch dtype to first create the embedder in.
        Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            bias: bool = True,
            dropout: float = 0.0,
            skip: bool = True,
            keep_dim: bool = False,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.__n_features = n_features
        self.bias = bias
        self.dropout = dropout
        self.skip = skip
        self.keep_dim = keep_dim
        self.drop = ptn.Dropout(dropout)
        self.project = ptn.Linear(
            in_features=n_features * mod_dim,
            out_features=n_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.score = ptn.Softmax(dim=-1)

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def n_features(self) -> int:
        """The number of features in the bag."""
        return self.__n_features

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.project.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.project.weight.dtype

    def importance(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Per-instance weights in the normed linear combination of features.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None
            Ignored for the :class:`InstanceWeightsMixer`.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the `inp` with the
            last dimension being dropped.

        """
        return self.score(self.project(inp.flatten(start_dim=-2)))

    def forward(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.
        mask: Tensor or None
            Ignored for the :class:`InstanceWeightsMixer`.

        Returns
        -------
        Tensor
            Depending on `keep_dim`, the output tensor has the same number of
            dimensions as the `inp` or one fewer. The next-to-last dimension
            is either 1 or dropped. The last dimension (of size `mod_dim`)
            contains the per-instance (normed) linear combination of all
            feature vectors.

        """
        if self.n_features == 0:
            return inp if self.keep_dim else inp.sum(dim=-2)
        out = self.drop(self.importance(inp).unsqueeze(dim=-2) @ inp)
        if self.skip:
            out += inp.mean(dim=-2, keepdim=True)
        return out if self.keep_dim else out.squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.project.reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        ActivatedMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.n_features,
            self.bias,
            self.dropout,
            self.skip,
            self.keep_dim,
            self.device,
            self.dtype,
        )
