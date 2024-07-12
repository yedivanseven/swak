from typing import Callable, Iterable
from .exceptions import SplitError
from ..magic import ArgRepr


class Split[S, T](ArgRepr):
    """Split an iterable of objects into two according to a decision criterion.

    Upon subclassing and/or instantiation, type annotation with the type of the
    elements in the iterator to be filtered and the return type of `wrapper`
    is recommended.

    Parameters
    ----------
    criterion: callable, optional
        The `condition` is called with one object at a time and must return a
        boolean value. Objects on which it evaluates to ``True` are sorted
        into one container and those where it evaluates to ``False`` end up
        in a second container. Defaults to ``None``, which returns the inherent
        truth values of the object in the iterable.
    wrapper: type or callable, optional
        If not given, an attempt will be made to determine the type of the
        container the callable instance was called with. Whether inferred or
        explicitly given, `wrapper` will be called twice, once with a list of
        a list of the elements that evaluated to ``True`` and once with those
        that evaluated to ``False``.

    """

    def __init__(
            self,
            criterion: Callable[[S], bool] | None = None,
            wrapper: type[T] | Callable[[list[S]], T] | None = None
    ) -> None:
        super().__init__(criterion, wrapper)
        self.criterion = criterion
        self.wrapper = wrapper

    def __call__(self, iterable: Iterable[S]) -> tuple[T, T]:
        """Split sequence into two according to cached decision criterion.

        Parameters
        ----------
        iterable: Iterable
            Objects to be split according to the cached (boolean) `criterion`.

        Returns
        -------
        tuple
            Two sequences of the same type as the input sequence or, if a
            `wrapper` was specified, of that type. Objects where the
            `criterion` evaluated to ``True`` are in the first return
            sequence and those evaluating to ``False`` are in the second.

        Raises
        ------
        SplitError
            If calling the criterion on any element of `iterable` raises
            an exception or if wrapping the results leads to an exception.

        """
        criterion = bool if self.criterion is None else self.criterion
        true = []
        false = []
        for i, element in enumerate(iterable):
            try:
                criterion_is_fulfilled = criterion(element)
            except Exception as error:
                msg = 'Error calling\n{}\non element #{}:\n{}\n{}:\n{}'
                name = self._name(criterion)
                err_cls = error.__class__.__name__
                fmt = msg.format(name, i, element, err_cls, error)
                raise SplitError(fmt)
            if criterion_is_fulfilled:
                true.append(element)
            else:
                false.append(element)
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            true = wrap(true)
            false = wrap(false)
        except Exception:
            msg = 'Could not wrap split results into instances of {}'
            raise SplitError(msg.format(self._name(wrap)))
        return true, false
