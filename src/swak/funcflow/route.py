from typing import Any, Self
from collections.abc import Iterator, Callable, Sequence, Iterable
from functools import singledispatchmethod
from ..misc import IndentRepr
from .exceptions import RouteError

type Call = type | Callable[..., Any]
type Calls = tuple[Call, ...]
type Routes = Sequence[int | Sequence[int]]


class Route[**P, T](IndentRepr):
    """Flexibly route arguments to a sequence of callables and collect results.

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that will be routed to the callables and a
    ``tuple`` specifying the concatenation of the return types of all
    callables,  ignoring empty tuples. If only a single object remains, the
    type of that object should be annotated.

    Parameters
    ----------
    routes: sequence of int or sequence of sequences of int, optional
        Specified as, e.g., ``[2, 0, 1]`` means that the first callable will be
        called with the third argument (index 2), the second with the first,
        and the third with the second. If callables take more than one
        argument, `routes` can be specified as ``[(2, 0), (), 1]``, which
        means that the first callable will be called with the third and first
        arguments, the second with no arguments, and the third with the second
        argument. Defaults to an empty tuple, meaning that no callables can be
        specified and, that, therefore, nothing is returned when calling the
        instance, no matter how many arguments it is called with.
    call: callable or iterable of callables, optional
        One callable or an iterator of callables that will be called with
        the arguments according to `routes`. Defaults to an empty tuple.
    *calls: callable
        Additional callables that will be called with the arguments according
        to `routes`. Together with `call`, there must be the same number
        of callables as there are routes.

    Raises
    ------
    RouteError
        If the `routes` cannot be parsed, if the number of `routes` does not
        match the number of callables specified with `call` and `calls`, or
        if (any of) `call` or any of `calls` is not, in fact, callable.

    """

    def __init__(
            self,
            routes: Routes = (),
            call: Call | Iterable[Call] = (),
            *calls: Call
    ) -> None:
        self.routes = self.__packed(routes)
        callables = self.__valid(call) + self.__valid(calls)
        self.calls = self.__compatible(*callables)
        routes = [r[0] if len(r) == 1 else r for r in self.routes]
        super().__init__(self.calls, routes)

    def __iter__(self) -> Iterator[Call]:
        # We could also iterate over instances of self ...
        return self.calls.__iter__()

    def __len__(self) -> int:
        return self.calls.__len__()

    def __bool__(self) -> bool:
        return bool(self.calls)

    def __contains__(self, item: Call) -> bool:
        return item in self.calls

    def __reversed__(self) -> Self:
        return self.__class__(
            list(reversed(self.routes)),
            *reversed(self.calls)
        )

    @singledispatchmethod
    def __getitem__(self, index: int) -> Call:
        # We could also return instances of self ...
        return self.calls[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(self.routes[index], *self.calls[index])

    def __hash__(self) -> int:
        return hash((self.calls, self.routes))

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.calls == other.calls and self.routes == other.routes
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.calls != other.calls or self.routes != other.routes
        return NotImplemented

    def __add__(self, other: Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(
                [*self.routes, *other.routes],
                *self.calls, *other.calls
            )
        return NotImplemented

    def __call__(self, *args: P.args) -> T:
        """Distribute arguments according to `routes` and forward to `calls`.

        Parameters
        ----------
        *args
            Arguments to be redistributed according to `routes` among `calls`.
            There must be at least `n_args` arguments. Extras will be ignored.

        Returns
        -------
        tuple or object
            Concatenation of all return values of all `calls` in order. If only
            one of the `calls` returns something other than an empty tuple,
            that object is returned.

        Raises
        ------
        RouteError
            If there are too few arguments to redistribute among `calls`
            according to `routes` or if calling one of the `calls` fails.

        """
        if (n_args := len(args)) < self.n_args:
            msg = 'Number of arguments must be at least {}, not {}!'
            raise RouteError(msg.format(self.n_args, n_args))
        results = []
        for i, (route, call) in enumerate(zip(self.routes, self)):
            try:
                result = call(*tuple(args[r] for r in route))
            except Exception as error:
                msg = '\n{} executing\n{}\nin route #{} {} of\n{}\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(err_cls, name, i, route, self, error)
                raise RouteError(fmt) from error
            else:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
        return results[0] if len(results) == 1 else tuple(results)

    @property
    def n_args(self) -> int:
        """The minimum number of arguments required for calling instances."""
        # Routes could be an empty tuple.
        if self.routes:
            # Each route could be an empty tuple.
            maxima = [max(route) for route in self.routes if route]
            # Maximum of the maximum integer in each route, if it exists.
            return max(maxima) + 1 if maxima else 0
        return 0

    @staticmethod
    def __valid(calls: Call | Iterable[Call]) -> tuple[Call, ...]:
        """Ensure that the argument is indeed an iterable of callables."""
        if callable(calls):
            return calls,
        iterable = True
        all_callable = False
        try:
            all_callable = all(callable(call) for call in calls)
        except TypeError:
            iterable = False
        if iterable and all_callable:
            return tuple(calls)
        raise RouteError('All paths in the route must be callable!')

    def __compatible(self, *calls: Call) -> Calls:
        """Check if the number of routes matches the number of callables."""
        if (n_routes := len(self.routes)) == (n_calls := len(calls)):
            return calls
        msg = 'Number of callables (={}) must match number of routes (={})!'
        raise RouteError(msg.format(n_calls, n_routes))

    @staticmethod
    def __packed(routes: Routes) -> tuple[tuple[int, ...], ...]:
        """Unify route specifications to tuples of integers."""
        tuples = []
        msg = 'Routes must be integers or tuples thereof, not {}'
        # Routes could be of completely wrong type, i.e., not iterable.
        try:
            safe_routes = [*routes]
        except TypeError as error:
            raise RouteError(msg.format(routes)) from error
        # If routes are iterable, elements could be ...
        for route in safe_routes:
            try:
                # ... some sort of sequence of integers ...
                tuples.append(tuple(int(r) for r in route))
            except TypeError:
                # ... or just a single integer.
                try:
                    tuples.append((int(route),))
                except TypeError as error:
                    raise RouteError(msg.format(route)) from error
            except ValueError as error:
                raise RouteError(msg.format(route)) from error
        return tuple(tuples)
