from typing import Any, Iterator, Callable, Self, Sequence
from functools import singledispatchmethod
from ..magic import IndentRepr
from .exceptions import RouteError

type Call = type | Callable[..., Any]
type Calls = tuple[Call, ...]
type Routes = Sequence[int | Sequence[int]]


class Route[**P, T](IndentRepr):
    """Flexibly route arguments to a sequence of callables and collect results.

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that will be routed to the callables and a ``tuple``
    specifying the concatenation of the return types of all callables, ignoring
    empty tuples. If only a single object remains, the type of that object
    should be annotated.

    Parameters
    ----------
    routes: sequence of int or sequence of sequences of int
        Specified as, e.g., ``[2, 0, 1]`` means that the first callable will be
        called with the third argument (index 2), the second with the first,
        and the third with the second. If callables take more than one
        argument, `routes` can be specified as ``[(2, 0), (0, 1), 1]``, which
        means that the first callable will be called with the third and first
        arguments, the second with the first and second, and the third with
        the second.
    *calls: callable
        Callable objects (functions, classes, etc.) that will be called with
        the arguments according to `routes`. There must be the same number
        of `calls` as there are routes.

    Properties
    ----------
    n_args: int
        The minimum number of arguments that instance must be called with.

    Raises
    ------
    RouteError
        If the `routes` cannot be parsed or if the number of `routes` does not
        match the number of `calls`.

    """

    def __init__(self, routes: Routes = (), *calls: Call) -> None:
        self.routes = self.__packed(routes)
        self.calls = self.__compatible(*calls)
        super().__init__(*calls)

    def __iter__(self) -> Iterator[Call]:
        # We could also iterate over instances of self ...
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __bool__(self) -> bool:
        return self.__len__() > 0

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

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Route):
            return self.calls == other.calls and self.routes == other.routes
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, Route):
            return self.calls != other.calls or self.routes != other.routes
        return NotImplemented

    def __add__(self, other: Self) -> Self:
        if isinstance(other, Route):
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
            one of the `calls` returns something other than an empty tuple, that
            object is returned.

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
                msg = 'Error executing\n{}\nin route #{} {} of\n{}\n{}:\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(name, i, route, self, err_cls, error)
                raise RouteError(fmt)
            else:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
        n_res = len(results)
        return tuple(results) if n_res == 0 or n_res > 1 else results[0]

    @property
    def n_args(self) -> int:
        """The number of arguments required for calling."""
        # Routes could be an empty tuple.
        if self.routes:
            # Each route could be an empty tuple.
            maxima = [max(route) for route in self.routes if route]
            # Maximum of the maximum integer in each route, if it exists.
            return max(maxima) + 1 if maxima else 0
        return 0

    def __compatible(self, *calls: Call) -> Calls:
        """Check if the number of routes matches the number of callables."""
        if (n_routes := len(self.routes)) == (n_calls := len(calls)):
            return calls
        msg = 'Number of callables (={}) must match the number of routes (={})!'
        raise RouteError(msg.format(n_calls, n_routes))

    @staticmethod
    def __packed(routes: Routes) -> tuple[tuple[int, ...], ...]:
        """Unify route specifications to tuples of integers."""
        tuples = []
        msg = 'Routes must be integers or tuples thereof, not {}'
        # Routes could be of completely wrong type, i.e., not iterable.
        try:
            safe_routes = [route for route in routes]
        except TypeError:
            raise RouteError(msg.format(routes))
        # If routes are iterable, elements could be ...
        for route in safe_routes:
            try:
                # ... some sort of sequence of integers ...
                tuples.append(tuple(int(r) for r in route))
            except TypeError:
                # ... or just a single integer.
                try:
                    tuples.append((int(route),))
                except TypeError:
                    raise RouteError(msg.format(route))
            except ValueError:
                raise RouteError(msg.format(route))
        return tuple(tuples)
