from typing import Any, Self
from collections.abc import KeysView, Callable, Iterator
from functools import reduce
from ast import literal_eval
import json
from json import JSONDecodeError
from pandas import Series
from .fields import Maybe
from .exceptions import (
    SchemaError,
    DefaultsError,
    ParseError,
    CastError,
    ValidationErrors
)

type Json = dict[str, Any]
type Raw = str | bytes | bytearray | Json | Series | None
type Schema = dict[str, type | Callable[[Any], Any]]
type Types = tuple[type, ...]
type Tried = tuple[Json, list[Exception]]


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
        'get',  # Do we even need this method?
        'keys',
        '_serialize'
    }

    __ignore_extra__ = False
    __raise_extra__ = True
    __respect_none__ = False

    def __new__(
            mcs,  # noqa: N804
            name: str,
            bases: Types,
            attrs: Json,
            **kwargs: Any
    ) -> 'SchemaMeta':
        # Create a new class object
        cls = super().__new__(mcs, name, bases, attrs)

        # Set behavior from past and present class keywords
        ancestral_variables = mcs.__ancestral(cls, '__dict__')
        cls.__ignore_extra__ = kwargs.pop(
            'ignore_extra',
            ancestral_variables.get('__ignore_extra__', mcs.__ignore_extra__)
        )
        cls.__raise_extra__ = kwargs.pop(
            'raise_extra',
            ancestral_variables.get('__raise_extra__', mcs.__raise_extra__)
        )
        cls.__respect_none__ = kwargs.pop(
            'respect_none',
            ancestral_variables.get('__respect_none__', mcs.__respect_none__)
        )

        # Consolidate values from "keys_from" keyword and the remaining kwargs
        kwargs = kwargs.pop('keys_from', {}) | kwargs

        # The schema is in the class __annotations__
        ancestral_schema = mcs.__ancestral(cls, '__annotations__')
        # Extract (string) keys from additional keyword arguments
        kwarg_schema = dict.fromkeys(kwargs, str)
        # Class body fields overwrite keyword fields overwrite inherited fields
        schema = ancestral_schema | kwarg_schema | cls.__annotations__
        # Validate the schema just assembled
        schema, schema_errors = mcs.__valid(name, schema, mcs.__blacklist__)

        # Ancestral defaults are in the class __defaults__
        ancestral_defaults = mcs.__ancestral(cls, '__defaults__')
        # Defaults for additional fields from keyword arguments are the keys
        kwarg_defaults = {k: k for k in kwargs if k not in cls.__annotations__}
        # Class variables overwrite keyword defaults overwrite ancestry
        updated_defaults = ancestral_defaults | kwarg_defaults | cls.__dict__
        # Only type-annotated class variables are relevant
        filtered_defaults = mcs.__filter(schema, updated_defaults)
        # Use __defaults__ because __dict__ cannot be set or updated directly
        defaults, default_errors = mcs.__tried(filtered_defaults, schema)

        # Raise accumulated errors, if any
        errors = schema_errors + default_errors
        if errors:
            raise ValidationErrors(name, errors)

        # Set hidden class variables
        cls.__blacklist__ = mcs.__blacklist__
        cls.__annotations__ = schema
        cls.__defaults__ = defaults

        return cls

    @staticmethod
    def __ancestral(descendant: 'SchemaMeta', attribute: str) -> Json:
        """Accumulate dictionary class variables down the inheritance tree."""
        # Get class ancestors starting with the oldest
        lineage = reversed(descendant.mro()[1:])
        # Accumulate inherited dictionary attributes, overwriting old with new
        return reduce(descendant.__merge(attribute), lineage, {})

    @staticmethod
    def __merge(attribute: str) -> Callable[[Json, Any], Json]:
        """Provide update function for dictionary class attributes."""

        def update(older: Json, newer: Any) -> Json:
            """Update dictionary attribute of parent with that of child."""
            return {**older, **getattr(newer, attribute, {})}

        return update

    @staticmethod
    def __filter(schema: Schema, defaults: Json) -> Json:
        """Filter down class __dict__ to keys present in the schema."""
        return {key: defaults[key] for key in defaults if key in schema}

    @staticmethod
    def __valid(
            name: str,
            schema: Schema,
            blacklist: set[str]
    ) -> tuple[Schema, list[Exception]]:
        """Validate that class-variable annotations are sane."""
        hidden = f'_{name}__'  # Pattern for double-underscore class variables
        errors = []
        for field, annotation in schema.items():
            if not callable(annotation):
                msg = f'Annotation of field "{field}" is not callable!'
                errors.append(SchemaError(msg))
            if field in blacklist:
                msg = f'Field "{field}" is on the blacklist {blacklist}!'
                errors.append(SchemaError(msg))
            if field.startswith(hidden):
                cleaned = field.removeprefix(hidden)
                msg = f'Field "__{cleaned}" starts with two underscores "__"!'
                errors.append(SchemaError(msg))
        return schema, errors

    @staticmethod
    def __tried(defaults: Json, schema: Schema) -> Tried:
        """Ensure that class-variable defaults are sane."""
        errors = []
        for item in defaults:
            # Check for None values and whether they are allowed
            default_is_none = defaults[item] is None
            type_is_not_maybe = not isinstance(schema[item], Maybe)
            if default_is_none and type_is_not_maybe:
                msg = (f'For the default value of field "{item} to be None,'
                       ' annotate it as Maybe(<YOUR_TYPE>) in the schema!')
                errors.append(DefaultsError(msg))
            # Check that schema annotations can be called on default values
            try:
                defaults[item] = schema[item](defaults[item])
            except (TypeError, ValueError):
                msg = (f'Default value for field "{item}" can'
                       ' not be cast to the desired type!')
                errors.append(DefaultsError(msg))
        return defaults, errors


# ToDo: Add polar-rs support!
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
    must also tolerate ``None`` values, which is realized by wrapping existing
    types into ``Maybe`` instances.

    The resulting object behaves in many ways like a dictionary, allowing
    dictionary-style, but also object-style access to data fields. Attributes
    of nested instances can be accessed dictionary-style (i.e., with the
    square-bracket accessor) with a dot.separated key.

    Parameters
    ----------
    mapping: dict, str, bytes, or Series, optional
        Dictionary with string keys, JSON string/bytes, or pandas Series.
        Defaults to an empty dictionary.
    **kwargs
        Can be any value or, for nested structures, again a dictionary with
        string keys or a JSON string/bytes or a pandas Series. Keyword
        arguments will override values already present in the `mapping`.

    Raises
    ------
    ValidationErrors
        ExceptionGroup containing any number of the following exceptions.
    ParseError
        If the (keyword) arguments cannot be parsed into a dictionary with
        string keys and if non-default fields are neither given in the
        `mapping` nor in the keyword arguments.
    CastError
        If the dictionary values cannot be cast into the types specified in
        the schema.

    Warnings
    --------
    This class is rather heavy, so do not use it to, e.g., wrap JSON payloads
    in high-throughput low-latency web services!

    See Also
    --------
    fields.Maybe

    """

    def __init__(self, mapping: Raw | Self = None, **kwargs: Any) -> None:
        # Fully nest the parsed and purged dictionaries with dot.separated keys
        parsed = self.__nest(self.__purge(self.__parse(mapping)))
        kwargs = self.__nest(self.__purge(self.__parse(kwargs, 1)))
        defaults = self.__nest(self.__defaults__)
        # Merge the fully nested dictionaries
        merged = self.__merge(defaults, self.__merge(parsed, kwargs))
        # Type-cast the merged dictionary
        cast = self.__cast(merged)
        # Set all dictionary items as object attributes
        self.__dict__.update(cast)

    def __getitem__(self, key: str) -> Any:
        # Raise for blacklisted keys
        if key in self.__blacklist__:
            raise KeyError(key)
        # Try and split the (string) key by dots
        try:
            root, *children = key.split('.')
        except AttributeError as error:
            cls = type(key).__name__
            msg = f'Keys must be strings, not {cls} like {key}!'
            raise KeyError(msg) from error
        # The key could also refer to an attribute like a property or a method
        try:
            value = self.__dict__.get(root, getattr(self, root))
        # ... but we still raise a KeyError to meet expectations
        except AttributeError as error:
            raise KeyError(key) from error
        # If the key contains dots, recurse down into the value
        return reduce(lambda x, y: x[y], children, value)

    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    def __str__(self) -> str:
        return json.dumps(self.__dict__, default=self._serialize)

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=4, default=self._serialize)

    __hash__ = None

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
        return self.__call__(other)

    def __ror__(self, other: Raw | Self) -> dict:
        try:
            return {**other, **self}
        except TypeError:
            return NotImplemented

    def __call__(self, mapping: Raw | Self = None, **kwargs: Any) -> Self:
        """Update one or more (nested) fields with `mapping` and kwargs.

        Parameters
        ----------
        mapping: dict, str, bytes, or Series, optional
            Dictionary with string keys, JSON string/bytes, or pandas Series.
            Defaults to an empty dictionary.
        **kwargs
            Can be any value or, for nested structures, again a dictionary with
            string keys or a JSON string/bytes or a pandas Series. Keyword
            arguments will override values already present in the `mapping`.

        Returns
        -------
        JsonObject
            A new instance of self with updated values.

        Raises
        ------
        ValidationErrors
            ExceptionGroup containing any number of the following exceptions.
        ParseError
            If the (keyword) arguments cannot be parsed into a dictionary with
            string keys.
        CastError
            If the dictionary values cannot be cast into the types specified in
            the schema.

        """
        # Fully nest the parsed and purged dictionaries with dot.separated keys
        parsed = self.__nest(self.__purge(self.__parse(mapping)))
        kwargs = self.__nest(self.__purge(self.__parse(kwargs, 1)))
        # Merge the call arguments
        merged = self.__merge(parsed, kwargs)
        # Left merge with self
        merged = self.__merge(self.__dict__, merged,True)
        # Instantiate a new, updated copy of self from the fully nested update
        return self.__class__(merged)

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

    # Do we even need this method?
    def get(self, item: str, default: Any = None) -> Any:
        """Get (nested) attribute by (dot.separated) name or default."""
        try:
            return self[item]
        except KeyError:
            return default

    def keys(self) -> KeysView[str]:
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
        # Try to parse the input as a JSON string ...
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
        except TypeError as error:
            # ... we're done with the recursion and simply return the input ...
            if level > 0:
                return obj
            # ... unless this was the initial, root-level call. Then we fail
            raise ParseError(f'Could not parse {obj} as JSON!') from error
        # Recurse further down into the value of the parsed dictionary
        return {key: self.__parse(parsed[key], level + 1) for key in parsed}

    def __purge(self, mapping: Json) -> Json:
        """Eliminate items with ``None`` value according to  `respect_none`."""
        filters = {True: lambda _: True, False: lambda xs: xs[1] is not None}
        return dict(filter(filters[self.__respect_none__], mapping.items()))

    @staticmethod
    def __stop_recursion_for(obj: Any) -> bool:
        """Criterion for stopping recursions dictionary nesting and merging.

        As we recursively traverse the tree of dictionary-like objects from
        root to leaves, we stop when we arrive at a leave that is no longer
        dictionary-like.

        """
        try:
            _ = [*obj.keys()]
        except (AttributeError, TypeError):
            return True
        return not hasattr(obj, '__getitem__')

    def __nest(self, mapping: Json | Self) -> Json:
        """Nest a dictionary with nesting implied by dot.separated keys."""
        # If the input is no longer dictionary-like, end the recursion
        if self.__stop_recursion_for(mapping):
            return mapping
        # If it is, initialize the return value ...
        result = {}
        # ... and iterate through the keys
        for key in mapping.keys():  # noqa: SIM118
            # Get the value to the current key
            value = mapping[key]
            # Depending on the type to key ...
            if isinstance(key, str):
                # ... split the root from the children
                root, *children = key.split('.')
            else:
                # ... or leave it as it is
                root, *children = key,
            # If the current key did have dots, ...
            if children:
                #  ... the value is elevated to a dict
                value = {'.'.join(children): value}
            # If the root key already exists in the results ...
            if root in result:
                # ... merge it with the new value
                result[root] = self.__merge(result[root], value)
            else:
                # ... or, if not, just set it to the new value
                result[root] = value
        # After nesting one level, recurse further down on the values
        return {key: self.__nest(value) for key, value in result.items()}

    def __merge(self, old: Json, new: Json, left: bool = False) -> Json:
        """Recursively deep-merge two dictionaries, outer or left."""
        if self.__stop_recursion_for(old) or self.__stop_recursion_for(new):
            return new
        # First the old values in order of appearance ...
        result = {key: old[key] for key in old if key not in new}
        # ... then intersection of old and new in order of appearance in old
        for key in [key for key in old if key in new]:
            result[key] = self.__merge(old[key], new[key], left)
        # If requested, add fields only present in new in order of appearance
        right = {} if left else {k: new[k] for k in new if k not in old}
        return result | right

    def __cast(self, mapping: Json) -> Json:
        """Cast all fields in the data structure to their specified type."""
        # Initialize accumulators
        cast = {}
        errors = []

        # Iterate over the fields in the schema
        for item, type_cast in self.__annotations__.items():
            try:
                value = mapping[item]
            except KeyError:
                msg = f'Missing non-default field "{item}"!'
                errors.append(ParseError(msg))
                continue
            try:
                cast[item] = type_cast(value)
            except (TypeError, ValueError):
                msg = f'Could not cast field "{item}" to the desired type!'
                errors.append(CastError(msg))
            except ValidationErrors as error_group:
                errors.append(error_group)
            if value is None and not isinstance(type_cast, Maybe):
                msg = (f'For the value of field "{item} to be None, annotate'
                       ' it as Maybe(<YOUR_TYPE>) in the schema!')
                errors.append(CastError(msg))

        # If we don't have to deal with extra fields, we're done
        if self.__ignore_extra__:
            if errors:
                raise ValidationErrors(self.__class__.__name__, errors)
            return cast

        # If not, first check if we even allow extra fields
        extra_fields = set(mapping) - set(self.__annotations__)
        if extra_fields and self.__raise_extra__:
            msg = f'Fields {extra_fields} are not in the schema!'
            errors.append(ParseError(msg))
            raise ValidationErrors(self.__class__.__name__, errors)

        # Even if extra fields are not ignored and are allowed, we need to ...
        for field in extra_fields:
            # ... check that their keys are strings, ...
            if not isinstance(field, str):
                msg = f'Extra field "{field}" does not have a string key!'
                errors.append(ParseError(msg))
                continue
            # ... check that their keys are not blacklisted, ...
            if field in self.__blacklist__:
                msg = (f'Extra field "{field}" is on the '
                       f'blacklist {self.__blacklist__}!')
                errors.append(ParseError(msg))
            # ... and check that their keys are valid python identifiers.
            if not all(part.isidentifier() for part in field.split('.')):
                msg = (f'Not all parts of the (potentially dot.separated) key'
                       f' of field "{field}" are valid python identifiers!')
                errors.append(ParseError(msg))

        # If we found anything fishy, raise all errors together
        if errors:
            raise ValidationErrors(self.__class__.__name__, errors)

        # Only now do we accept and merge extra fields.
        extras = {field: mapping[field] for field in extra_fields}
        return {**cast, **extras}
