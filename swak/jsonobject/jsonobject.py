from typing import Callable, Any, Iterator, Self
from functools import reduce
from collections import defaultdict
from ast import literal_eval
import json
from json import JSONDecodeError
from pandas import Series
from .fields import Maybe
from .exceptions import SchemaError, DefaultsError, ParseError, CastError

type Json = dict[str, Any]
type Raw = str | bytes | bytearray | Json | Series | None
type Schema = dict[str, type | Callable[[Any], Any]]
type Types = tuple[type, ...]


class SchemaMeta(type):
    """Metaclass parsing, validating, and setting data schema definitions.

    This class is not intended to be instantiated directly but only to be used
    as a metaclass! When used as such, it parses type-annotated class variables
    (with or without default values) into `schema` and `defaults` class
    variables, and checks that type annotations are callable, can be called
    on the default values, and that default values are not ``None`` unless
    wrapped with ``Maybe``. Additionally, it makes sure that schema fields
    cannot collide with existing class variables as well as instance attributes
    and methods.

    """

    __blacklist__ = {
        'as_json',
        'as_series',
        'as_dtype',
        'get',
        'keys',
        'ignore_extra',
        'raise_extra',
        'respect_none',
        '_serialize'
    }

    ignore_extra = False
    raise_extra = True
    respect_none = False

    def __new__(
            mcs,
            name: str,
            bases: Types,
            attrs: Json,
            **kwargs: Any
    ) -> 'SchemaMeta':
        cls = super().__new__(mcs, name, bases, attrs)

        # Keywords passed on inheriting from an instance of this metaclass
        cls.ignore_extra = kwargs.pop('ignore_extra', mcs.ignore_extra)
        cls.raise_extra = kwargs.pop('raise_extra', mcs.raise_extra)
        cls.respect_none = kwargs.pop('respect_none', mcs.respect_none)

        # The schema is in the class __annotations__
        raw_schema = mcs.__get(cls, '__annotations__')
        # Extract (string) keys from additional keyword arguments
        kwarg_schema = {key: str for key in kwargs}
        # Fields defined in the class body overwrite keyword fields
        schema = kwarg_schema | raw_schema
        cls.__schema__ = mcs.__validate_schema(name, schema, mcs.__blacklist__)

        # Default values for class variables are in the class __dict__
        raw_defaults = mcs.__filter(cls, mcs.__get(cls, '__dict__'))
        # Defaults for additional fields from keyword arguments are the keys
        kwarg_defaults = {key: key for key in kwargs if key not in raw_schema}
        # Fields defined in the class body overwrite keyword fields
        defaults = kwarg_defaults | raw_defaults
        cls.__defaults__ = mcs.__validate_defaults(defaults, cls.__schema__)

        # JSON fields that conflict with methods or properties are forbidden
        cls.__blacklist__ = mcs.__blacklist__

        return cls

    @staticmethod
    def __get(cls: 'SchemaMeta', attribute: str) -> Json:
        """Accumulate dictionary class variables down the inheritance tree."""
        # Get class ancestors starting with the oldest
        lineage = reversed(cls.mro())
        # Accumulate inherited dictionary attributes, overwriting old with new
        return reduce(cls.__merge(attribute), lineage, {})

    @staticmethod
    def __merge(attribute: str) -> Callable[[Json, Any], Json]:
        """Provide update function for dictionary class attributes."""

        def update(older: Json, newer: Any) -> Json:
            """Update dictionary attribute of parent with that of child."""
            return {**older, **getattr(newer, attribute, {})}

        return update

    @staticmethod
    def __filter(cls: 'SchemaMeta', defaults: Json) -> Json:
        """Filter down class __dict__ to keys present in the schema."""
        return {key: defaults[key] for key in defaults if key in cls.__schema__}

    @staticmethod
    def __validate_schema(
            name: str,
            schema: Schema,
            blacklist: set[str]
    ) -> Schema:
        """Validate that class-variable annotations are sane."""
        msg = ''
        hidden = f'_{name}__'
        if not_callable := not all(callable(val) for val in schema.values()):
            msg += '\nAll schema annotations must be callable!'
        if any_forbidden := any(key in blacklist for key in schema.keys()):
            msg += f'\nField must not include any of {blacklist}!'
        if any_hidden := any(key.startswith(hidden) for key in schema.keys()):
            msg += '\nField names must not start with "__"!'
        if any([not_callable, any_forbidden, any_hidden]):
            raise SchemaError(msg)
        return schema

    @staticmethod
    def __validate_defaults(defaults: Json, schema: Schema) -> Json:
        """Validate that class-variable defaults are sane."""
        none_defaults = []
        none_msg = ''
        cast_defaults = []
        cast_msg = ''
        for item in defaults:
            # Check None values
            default_is_none = defaults[item] is None
            type_is_not_maybe = not isinstance(schema[item], Maybe)
            if default_is_none and type_is_not_maybe:
                none_defaults.append(item)
                none_msg = (f'\nFor defaults {none_defaults} to be None, mark'
                            ' them as Maybe(<YOUR_TYPE>) in the schema!')
            # Check can cast values
            try:
                defaults[item] = schema[item](defaults[item])
            except (TypeError, ValueError):
                cast_defaults.append(item)
                cast_msg = (f'\nDefaults {cast_defaults} can not'
                            ' be cast to the desired types!')
        if any([none_defaults, cast_defaults]):
            msg = ''.join([none_msg, cast_msg])
            raise DefaultsError(msg)
        return defaults


class JsonObject(metaclass=SchemaMeta):
    """Flexible Dataclass-like data structure with enforced type schema.

    This class is not meant to ever be instantiated directly. Rather,
    inherit from it, and specify fields as type-annotated class variables,
    potentially giving also default values. Values for non-default fields
    must be provided on instantiation in the form of a JSON string, a
    dictionary-like object, or keyword arguments. The handling of additional
    fields can be specified via boolean class keywords `ignore_extra`
    (defaults to ``False``) and `raise_extra` (defaults to ``True``).

    By default, JSON fields with a ``None`` value are ignored and treated as
    not being present. To actually set fields to ``None`` (and, potentially,
    overwrite defaults), the class keyword `respect_none` needs to be set
    to ``True`` on subclass definition. Note, however, that type annotations
    must also tolerate ``None`` values. This can be achieved by using
    custom types or by wrapping existing types into ``Maybe`` instance.

    The resulting object behaves in many ways like a dictionary, allowing
    dictionary-style, but also object-style access to data fields. Calling the
    object with a JSON string, dictionary, and/or keyword arguments yields
    a new instance with updated data.

    Parameters
    ----------
    mapping: dict, str, bytes, or Series, optional
        Dictionary with string keys, JSON string/bytes, or pandas Series.
        Defaults to an empty dictionary.
    **kwargs
        Can be any value or, for nested structures, again a dictionary with
        string keys or a JSON string/bytes or a pandas Series. Keywords will
        override values already present in the `mapping`.

    Raises
    ------
    ParseError
        If the (keyword) arguments cannot be parsed into a dictionary with
        string keys.
    CastError
        If the dictionary values cannot be cast into the types specified in
        the schema.

    """

    def __init__(self, mapping: Raw | Self = None, **kwargs: Any) -> None:
        # Flatten parsed and purged dictionaries into dot.separated key format
        flat_parsed = self.__flatten(self.__purge(self.__parse(mapping)))
        flat_kwargs = self.__flatten(self.__purge(self.__parse(kwargs, 1)))
        flat_defaults = self.__flatten(self.__defaults__)
        # Merge flattened dictionaries
        merged = {**flat_defaults, **flat_parsed, **flat_kwargs}
        # Type-cast merged dictionary after nesting them again
        cast = self.__cast(self.__nest(merged))
        # Set dictionary items as object attributes
        self.__dict__.update(cast)

    def __getitem__(self, key: str) -> Any:
        # Try and split the (string) key by dots
        try:
            root, *children = key.split('.')
        except AttributeError:
            cls = type(key).__name__
            raise KeyError(f'Keys must be strings, not {cls} like {key}!')
        # The key could also refer to an attribute like a property or a method
        value = self.__dict__.get(root, self.__getattribute__(root))
        # If the key contains dots, recurse down into the value
        return reduce(lambda x, y: x[y], children, value)

    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    def __str__(self) -> str:
        return json.dumps(self.__dict__, default=self._serialize)

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=4, default=self._serialize)

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ != other.__dict__
        return NotImplemented

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __bool__(self) -> bool:
        return bool(self.__dict__)

    def __or__(self, other: Raw | Self) -> Self:
        return self(other)

    def __call__(self, mapping: Raw | Self = None, **kwargs: Any) -> Self:
        """Update one or more (nested) fields with `mapping` and kwargs.

        Parameters
        ----------
        mapping: dict, str, bytes, or Series, optional
            Dictionary with string keys, JSON string/bytes, or pandas Series.
            Defaults to an empty dictionary.
        **kwargs
            Can be any value or, for nested structures, again a dictionary with
            string keys or a JSON string/bytes or a pandas Series. Keywords will
            override values already present in the `mapping`.

        Returns
        -------
        JsonObject
            A new instance of self with updated values.

        Raises
        ------
        ParseError
            If the (keyword) arguments cannot be parsed into a dictionary with
            string keys and if non-default fields are neither given in the
            `mapping` nor in the keyword arguments.
        CastError
            If the dictionary values cannot be cast into the types specified in
            the schema.

        """
        # Flatten parsed and purged dictionaries into dot.separated key format
        flat_parsed = self.__flatten(self.__purge(self.__parse(mapping)))
        flat_kwargs = self.__flatten(self.__purge(self.__parse(kwargs, 1)))
        flat_self = self.__flatten(self)
        # Merge flattened dictionaries
        merged = {**flat_parsed, **flat_kwargs}
        # Only keep those fields in the update that are already present
        update = {key: merged.get(key, flat_self[key]) for key in flat_self}
        # Instantiate a new, updated copy of self from the fully nested update
        return self.__class__(self.__nest(update))

    @property
    def as_json(self) -> Json:
        """JSON-serializable dictionary representation."""
        return json.loads(str(self))

    @property
    def as_dtype(self) -> str:
        """Representation in a cell of a pandas data frame."""
        return self.__str__()

    @property
    def as_series(self) -> Series:
        """Representation as a pandas series."""
        data = {key: getattr(self[key], 'as_dtype', self[key]) for key in self}
        name = self.__class__.__name__
        return Series(data, name=name)

    def get(self, item: str, default: Any = None) -> Any:
        """Get attributes by name. Return `default` if name does not exist."""
        return self.__dict__.get(item, default)

    def keys(self):
        """Attribute names as dictionary keys."""
        return self.__dict__.keys()

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Default JSON-encoding for attributes not trivially serializable."""
        return obj.as_json if hasattr(obj, 'as_json') else repr(obj)

    def __parse(self, obj: Raw | Self, level: int = 0) -> Json:
        """Recursively parse input into a (nested) dictionary."""
        # For the initial, root-level call, None means an empty dictionary
        mapping = {} if obj is None and level == 0 else obj
        # Try to parse input as a JSON string ...
        try:
            parsed = json.loads(mapping)
        except (TypeError, JSONDecodeError):
            # ... or some other string representation of a python object
            try:
                parsed = literal_eval(mapping)
            # In case of failure, it might already be a dictionary
            except (TypeError, ValueError, SyntaxError):
                parsed = mapping
        # If it is, this should work.
        try:
            parsed = {**parsed}
        # If not ...
        except TypeError:
            # ... we're done with the recursion and simply return the input ...
            if level > 0:
                return obj
            # ... unless this was the initial, root-level call. Then we fail
            raise ParseError(f'Could not parse {obj} as JSON!')
        # Recurse further down into the value of the parsed dictionary
        return {key: self.__parse(parsed[key], level + 1) for key in parsed}

    def __purge(self, mapping: Json) -> Json:
        """Eliminate items with ``None`` value according to  `respect_none`."""
        filters = {True: lambda _: True, False: lambda xs: xs[1] is not None}
        return dict(filter(filters[self.respect_none], mapping.items()))

    @staticmethod
    def __stop_recursion_for(obj: Any) -> bool:
        """Criterion for stopping recursions in flattening and nesting.

        As we recursively traverse the tree of dictionary-like object from
        root to leaves, we stop when we arrive at a leave that (a) is no
        longer dictionary-like or (b) still is a dictionary, but has non-string
        keys (which a JSON can't have), or (c) when it is an empty dictionary.

        """
        return (
            not hasattr(obj, 'keys') or
            any(type(key) is not str for key in obj.keys()) or
            len(obj.keys()) == 0
        )

    def __nest(self, mapping: Json | Self) -> Json:
        """Nest a dictionary with nesting implied by dot.separated keys."""
        # If the input has no "keys" method or keys are not str, end recursion
        if self.__stop_recursion_for(mapping):
            return mapping
        # If the input still has a "keys" method, initialize the return value
        result = defaultdict(dict)
        # Iterate through the keys
        for key in mapping.keys():
            # Split root-level key from the children
            root, *children = key.split('.')
            # Get the value to the current key
            value = mapping[key]
            # If the current key has dots, add child with dot-separated key
            if children:
                result[root].update({'.'.join(children): value})
            # If the current node is already a leave, just set its value
            else:
                result[root] = value
        # After nesting one level, recurse further down on the values
        return {key: self.__nest(value) for key, value in result.items()}

    def __flatten(self, mapping: Json | Self) -> Json:
        """Flatten a nested into a flat dictionary with dot.separated keys."""
        # If the input has no "keys" method or keys are not str, end recursion
        if self.__stop_recursion_for(mapping):
            return mapping
        # If it is, initialize return value
        result = {}
        # Iterate through the keys.
        for key in mapping.keys():
            # Get the value to the current key
            value = mapping[key]
            # If the value has no "keys" method or keys are not str, stop
            if self.__stop_recursion_for(value):
                result[key] = value
            # If the value still is dictionary-like, iterate over its keys
            else:
                for child in value.keys():
                    # Append child key to current key
                    child_key = '.'.join([key, child])
                    # Get value of child.
                    child_value = value[child]
                    # Update results with recursively resolved leaves
                    result.update(self.__flatten({child_key: child_value}))
        return result

    def __cast(self, mapping: Json) -> Json:
        """Cast all fields in the data structure to their specified type."""
        cast = {}
        uncastable = []
        uncastable_msg = ''
        missing = []
        missing_msg = ''
        for item, type_cast in self.__schema__.items():
            try:
                cast[item] = type_cast(mapping[item])
            except (TypeError, ValueError):
                uncastable.append(item)
                uncastable_msg = f'Could not cast JSON fields {uncastable}!'
            except KeyError:
                missing.append(item)
                missing_msg = f'Missing non-default fields {missing}!'
        if uncastable:
            raise CastError(uncastable_msg)
        if missing:
            raise ParseError(missing_msg)
        if self.ignore_extra:
            return cast
        extra_fields = set(mapping) - set(self.__schema__)
        if extra_fields and self.raise_extra:
            raise ParseError(f'Fields {extra_fields} are not in schema!')
        if any(key in self.__blacklist__ for key in extra_fields):
            msg = f'Extra fields must not include any of {self.__blacklist__}!'
            raise ParseError(msg)
        extras = {field: mapping[field] for field in extra_fields}
        return {**cast, **extras}
