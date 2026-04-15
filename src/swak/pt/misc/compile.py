from typing import Any
import torch as pt
from ..types import Module
from ..exceptions import CompileError


class Compile:
    """Partial of the ``compile`` top-level function or Module method.

    Parameters
    ----------
    inplace: bool, optional
        Whether to compile the model in place (by calling its ``compile``
        method) or create a new, compiled instance. Defaults to ``True``
    model: Module, optional
        For convenience, the model to compile can already be given at
        instantiation. However, Nothing will happen until instances are called.
    **kwargs
        Additional keyword arguments are forwarded to the ``compile`` function
        or method call. See the `Documentation <https://pytorch.org/docs/
        stable/generated/torch.compile.html#torch-compile>`__ for details.

    """

    def __init__(
            self,
            inplace: bool = True,
            model: Module | None = None,
            **kwargs: Any
    ) -> None:
        self.inplace = inplace
        self.model = model
        self.kwargs = kwargs

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        model = None if self.model is None else 'model'
        kwargs = (f'{k}={v!s}' for k, v in self.kwargs.items())
        kwargs = ', '.join(kwargs)
        sep = ', ' if kwargs else ''
        return f'{cls}({self.inplace}, {model}{sep}{kwargs})'

    def __call__(self, model: Module | None = None, **kwargs: Any) -> Module:
        """Compile a Module with the given options.

        Parameters
        ----------
        model: Module, optional
            If no model was given on instantiation, one must be given here.
            Otherwise, there would be nothing to compile. If a model was given
            on instantiation and one is given here, the latter replaces the
            former. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into those given at
            instantiation and then forward to the ``compile`` function
            or method call. See the `Documentation <https://pytorch.org/docs/
            stable/generated/torch.compile.html#torch-compile>`__ for details.

        Returns
        -------
        Module
            The compiled module.

        Raises
        ------
        CompileError
            If no model was given, neither at instantiation, nor when calling
            instances.

        """
        model = self.model if model is None else model
        if model is None:
            raise CompileError('No model to compile!')
        merged_kwargs = self.kwargs | kwargs
        if self.inplace:
            model.compile(**merged_kwargs)
            return model
        return pt.compile(model, **merged_kwargs)
