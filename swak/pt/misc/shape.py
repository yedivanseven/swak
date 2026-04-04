"""Convenient classes (and functions) that do not fit any other category."""

from typing import Any, Self, overload
from collections.abc import Iterable
from functools import cached_property, singledispatchmethod
from itertools import chain
import torch as pt
from ...misc import ArgRepr
from ..types import Tensor, Tensors
from ..exceptions import ShapeError, DeviceError, DTypeError, ValidationErrors


class Stack(ArgRepr):
    """Simple partial of PyTorch's top-level `stack` function.

    Parameters
    ----------
    dim: int, optional
        The new dimension along which to stack the tensors. Defaults to 0.

    """

    def __init__(self, dim: int = 0) -> None:
        super().__init__(dim)
        self.dim = dim

    def __call__(self, tensors: Tensors | list[Tensor]) -> Tensor:
        """Concatenate the given tensors along one of their dimensions.

        Parameters
        ----------
        tensors: tuple or list of tensors
            Tensors to stack along a new dimension.

        Returns
        -------
        Tensor
            The stacked tensors.

        """
        return pt.stack(tensors, dim=self.dim)


class Cat(ArgRepr):
    """Simple partial of PyTorch's top-level `cat` function.

    Parameters
    ----------
    dim: int, optional
        The dimension along which to concatenate the tensors. Defaults to 0.

    """

    def __init__(self, dim: int = 0) -> None:
        super().__init__(dim)
        self.dim = dim

    def __call__(self, tensors: Tensors | list[Tensor]) -> Tensor:
        """Concatenate the given tensors along one of their dimensions.

        Parameters
        ----------
        tensors: tuple or list of tensors
            Tensors to concatenate along an existing dimension.

        Returns
        -------
        Tensor
            The concatenated tensors.

        """
        return pt.cat(tensors, dim=self.dim)


class LazyCatDim0:
    """Lazily concatenate a sequence of tensors along their first dimension.

    Concatenating a large number of even small tensors (or a small number of
    large tensors) causes a memory spike because, temporarily, two copies of
    all tensors are needed. Sometimes, this simply cannot be avoided.
    However, when only a small part of the full concatenation of all tensors is
    needed at any given time, e.g., when chopping off micro-batches of training
    data to feed to a model, the present class provides an alternative:
    Constituent tensors are kept as is and concatenation is only performed when
    slices or element(s) along the first dimension are requested. These are
    selected from the constituents first and only then concatenated (and,
    thus, copied). Slicing and element selection of further dimensions
    is delayed until selection and concatenation along the first dimension is
    completed.

    Parameters
    ----------
    tensors: iterable
        The iterable of tensors to cache, all of which must have the same
        number of dimensions and the exact same sizes in all dimensions but
        the first.

    Raises
    ------
    ShapeError
        If any tensor is a scalar, that is, has zero dimensions, or if the
        shape after the first dimension is not the same across all tensors.
        Also, if there are no tensors to wrap.
    DeviceError
        If tensors are spread over multiple devices.
    DTypeError
        If tensors have multiple dtypes.

    """

    def __init__(self, tensors: Iterable[Tensor]) -> None:
        self.__tensors = self._valid(tuple(tensors))

    @staticmethod
    def _valid(tensors: Tensors) -> Tensors:
        """Run a few validations on the homogeneity of the wrapped tensors."""
        if not tensors:
            msg = 'Expected a non-empty iterable of tensors!'
            raise ShapeError(msg)
        errors = []
        if any(tensor.dim() == 0 for tensor in tensors):
            msg = 'Scalar tensors can not be concatenated!'
            errors.append(ShapeError(msg))
        _, *shape = tensors[0].shape
        if any(list(tensor.shape[1:]) != shape for tensor in tensors[1:]):
            msg = 'All tensors must be of shape {} after the first dimension!'
            errors.append(ShapeError(msg.format(shape)))
        if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
            msg = 'All tensors must be on the same device!'
            errors.append(DeviceError(msg))
        if any(tensor.dtype is not tensors[0].dtype for tensor in tensors[1:]):
            msg = 'All tensors must have the same dtype!'
            errors.append(DTypeError(msg))
        if errors:
            raise ValidationErrors('Validation failed', errors)
        return tensors

    @cached_property
    def lookup(self) -> tuple[tuple[int, int], ...]:
        """A lazily computed and cached lookup table for indices."""
        return tuple(
            (i, j)
            for i, tensor in enumerate(self.__tensors)
            for j in range(tensor.size(0))
        )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={len(self.lookup)})'

    def __len__(self) -> int:
        return len(self.lookup)

    def __iter__(self) -> chain[Tensor]:
        return chain.from_iterable(self.__tensors)

    def __contains__(self, elem: Any) -> bool:
        return any(elem in tensor for tensor in self.__tensors)

    @singledispatchmethod
    def __getitem__(self, index) -> Tensor:
        idx, elem = self.lookup[index]
        return self.__tensors[idx][elem]

    @__getitem__.register
    def _(self, index: slice) -> Tensor:
        return pt.cat([
            self.__tensors[idx][elem:elem + 1]
            for idx, elem
            in self.lookup[index.start:index.stop]
        ])[::index.step]

    @__getitem__.register
    def _(self, index: list) -> Tensor:
        return pt.cat([
            self.__tensors[idx][elem:elem + 1]
            for idx, elem
            in (self.lookup[idx] for idx in index)
        ])

    @__getitem__.register
    def _(self, index: tuple) -> Tensor:
        if not index:
            return self[:]
        idx, *indices = index
        match idx:
            case slice():
                return self[idx][:, *indices]
            case int():
                return self[idx][*indices]


    @property
    def dtype(self) -> pt.dtype:
        """The common dtype of all cached tensors."""
        return self.__tensors[0].dtype

    @property
    def device(self) -> pt.device:
        """The common device of all cached tensors."""
        return self.__tensors[0].device

    @cached_property
    def shape(self) -> pt.Size:
        """The shape of the full concatenation of the wrapped tensors."""
        _, *shape = self.__tensors[0].shape
        return pt.Size([len(self), *shape])

    @overload
    def size(self, dim: int) -> int:
        ...

    @overload
    def size(self, dim: None = None) -> pt.Size:
        ...

    def size(self, dim = None):
        """The size of the full concatenation of the wrapped tensors.

        Parameters
        ----------
        dim: int, optional
            The dimension for which to return the size. Defaults to ``None``.
            If not given the `shape` of the full concatenation of all cached
            tensors is returned.

        Returns
        -------
        int
            If the size along a certain dimension was requested.
        Size
            A PyTorch ``Size`` object specifying the `shape` of the full
            concatenation of all cached tensors.

        """
        return self.shape[dim] if dim is not None else self.shape

    def to(self, *args: Any) -> Self:
        """Returns a new instance of self, wrapping transformed tensors.

        See the PyTorch `documentation <https://pytorch.org/docs/stable/
        generated/torch.Tensor.to.html#torch-tensor-to>`_ for possible call
        signatures, but keep in mind that all listed operations will create
        a copy of the wrapped tensors after all.

        """
        return self.__class__(tensor.to(*args) for tensor in self.__tensors)
