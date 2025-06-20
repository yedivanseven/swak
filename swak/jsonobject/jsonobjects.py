from typing import Any, Self
from collections.abc import Iterator, Iterable
from functools import singledispatchmethod
import json
from json.decoder import JSONDecodeError
from ast import literal_eval
from pandas import Series, DataFrame
from .exceptions import ParseError, SchemaError, CastError, ValidationErrors
from .jsonobject import SchemaMeta, JsonObject

type Json = dict[str, Any]
type Record = str | bytes | bytearray | Json | Series | JsonObject | None
type Records = str | bytes | bytearray | DataFrame | Record | Iterable[Record]


# ToDo: Add polar-rs support!
class JsonObjects[T]:
    """List-like container for JSON-serializable dictionaries.

    This class is not meant to ever be instantiated directly. Rather,
    inherit from it, and specify a subclass of ``JsonObject`` with the
    `item_type` class keyword on sub-classing.

    Parameters
    ----------
    items : list, optional
        List of JSON-serializable dictionaries. Defaults to an empty list.
    *args
        Additionally, any number of JSON-serializable dictionaries with the
        schema specified by the `item_type` can be provided. These will be
        appended to the `items`.

    Raises
    ------
    ParseError
        If the constructor argument(s) can not be parsed as a list of
        dictionaries.
    SchemaError
        If `item_type` is not a subclass of ``JsonObject``.

    Warnings
    --------
    This class is rather heavy, so do not use it to, e.g., wrap JSON payloads
    in high-throughput low-latency web services!

    """

    def __init__(self, items: Records | Self = (), *args: Record) -> None:
        parsed = self.__parse(items)
        itemized = self.__itemize(parsed)
        self.__items = self.__wrap([*itemized, *args])

    def __init_subclass__(cls, **kwargs: Any) -> None:
        try:
            item_type = kwargs.pop('item_type')
        except KeyError:
            try:
                item_type = cls.mro()[1].__item_type__
            except AttributeError as error:
                msg = ('If not inherited, an "item_type" must be defined as'
                       'a keyword argument in the class call on definition!')
                raise SchemaError(msg) from error
        cls.__item_type__ = cls.__class_checked(item_type)
        super().__init_subclass__(**kwargs)

    def __str__(self) -> str:
        return json.dumps(self.__items, default=lambda obj: obj.as_json)

    def __repr__(self) -> str:
        return json.dumps(self.__items, indent=4, default=lambda o: o.as_json)

    def __iter__(self) -> Iterator[T]:
        return self.__items.__iter__()

    def __reversed__(self) -> Self:
        return self.__class__(reversed(self.__items))

    def __len__(self) -> int:
        return self.__items.__len__()

    def __getattr__(self, key: str) -> list:
        missing = object()
        values = [item.get(key, missing) for item in self.__items]
        if all(value is missing for value in values):
            cls = self.__class__.__name__
            msg = f"'{cls}' object has no attribute '{key}'"
            raise AttributeError(msg)
        return [None if value is missing else value for value in values]

    @singledispatchmethod
    def __getitem__(self, index: int) -> T:
        return self.__items[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(*self.__items[index])

    @__getitem__.register
    def _(self, key: str) -> list:
        missing = object()
        values = [item.get(key, missing) for item in self.__items]
        if all(value is missing for value in values):
            raise KeyError(key)
        return [None if value is missing else value for value in values]

    def __bool__(self) -> bool:
        return bool(self.__items)

    def __contains__(self, other: Record) -> bool:
        try:
            return self.__item_type__(other) in self.__items
        except (ParseError, CastError, ValidationErrors):
            return False

    __hash__ = None

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.__items == other.__items
        return NotImplemented

    def __ne__(self: T, other: T) -> bool:
        if isinstance(other, self.__class__):
            return self.__items != other.__items
        return NotImplemented

    def __add__(self, others: Records | Self) -> Self:
        return self.__class__(self, *self.__class__(others))

    def __radd__(self, others: Records | Self) -> Self:
        return self.__class__(self.__class__(others), *self)

    def __call__(self, mapping: Record = None, **kwargs: Any) -> Self:
        """Update one or more (nested) fields within each item.

        Parameters
        ----------
        mapping: dict or str, optional
            Dictionary with string keys, JSON string/bytes, or pandas Series.
            Defaults to an empty dictionary.
        **kwargs:
            Can be any value or, for nested structures, again a dictionary with
            string keys or a JSON string/bytes or a pandas Series. Keyword
            arguments will override values already present in the `mapping`.

        Returns
        -------
        JsonObjects
            A new instance of self with updated fields in each item.

        Raises
        ------
        ParseError
            If the (keyword) arguments cannot be parsed into a dictionary with
            string keys.
        CastError
            If the dictionary values cannot be cast into the types specified in
            the schema of the `item_type`.

        """
        return self.__class__(item(mapping, **kwargs) for item in self)

    @property
    def as_json(self) -> list[Json]:
        """JSON-serializable representation."""
        return [item.as_json for item in self.__items]

    @property
    def as_dtype(self) -> str:
        """Representation in the cell of a pandas data frame."""
        return self.__str__()

    @property
    def as_df(self) -> DataFrame:
        """Representation as a pandas data frame."""
        data = [item.as_series for item in self]
        if data:
            columns = None
        else:
            columns = list(self.__item_type__.__annotations__.keys())
        df = DataFrame(data, columns=columns)
        df.columns.name = self.__item_type__.__name__
        return df.reset_index(drop=True)

    @staticmethod
    def __class_checked(item_type: type[JsonObject]) -> type[JsonObject]:
        """Allow only JsonObject and JsonObjects as item_type."""
        right_type = isinstance(item_type, SchemaMeta)
        right_class = issubclass(item_type, JsonObject)
        if right_type and right_class:
            return item_type
        raise SchemaError('item_type must be a subclass of JsonObject!')

    @staticmethod
    def __parse(items: Records) -> list[Record]:
        """Parse input into a list of something."""
        # Define parsers for converting input into a list of items.
        parsers = (
            lambda x: json.loads(x),                # JSON string
            lambda x: literal_eval(x),              # Some other string
            lambda x: x.to_dict(orient='records'),  # Dataframe
            lambda x: [] if x is None else x        # None or some other object
        )
        # Try parsers one after another
        for parse in parsers:
            try:
                parsed = parse(items)
            except (
                JSONDecodeError,  # json.loads
                TypeError,        # json.loads and literal_eval
                ValueError,       # literal_eval
                SyntaxError,      # literal_eval
                AttributeError    # Dataframe
            ):
                continue
            else:
                return parsed
        raise ParseError(f'Could not parse {items} as JSON!')

    @staticmethod
    def __itemize(items: list[Record]) -> list[Json]:
        """Convert list of something into a list of JSONs."""
        # Define patterns that convert input to a list of dicts
        patterns = (
            lambda x: [{**x}],  # List of dicts
            lambda x: [*x]      # List of iterable, possibly a string
        )
        # First, try list of dicts, then list of anything
        for pattern in patterns:
            try:
                itemized = pattern(items)
            except TypeError:
                continue
            else:
                return itemized
        raise ParseError(f'Could not parse {items} as JSON!')

    def __wrap(self, items: list[Record]) -> list[T]:
        """Cast each item in a list of JSONs to the item schema."""
        errors = []
        wrapped = []
        for item in items:
            try:
                wrapped.append(self.__item_type__(item))
            except (ParseError, CastError, ValidationErrors) as error:
                errors.append(error)
        if errors:
            raise ValidationErrors(self.__class__.__name__, errors)
        return wrapped
