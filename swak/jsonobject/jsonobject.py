
from typing import Callable, Any
from functools import reduce
from .fields import Maybe
from .exceptions import SchemaError, DefaultsError

type Json = dict[str, Any]


class SchemaMeta(type):
    """Metaclass parsing, validating, and setting data schema definitions.

    This class is not intended to be instantiated directly but only to be used
    as a metaclass! When used as such, it parses type-annotated class variables
    (with or without default values) into `schema` and `defaults` class
    variables, and checks that type annotations are callable, can be called
    on the default values, and that default values are not ``None`` unless
    wrapped with ``Maybe``. Additionally, we make sure that schema fields
    cannot collide with existing class variables as well as instance attributes
    and methods.

    """

    __blacklist__ = {
        'as_json',
        'as_df',
        'as_dtype',
        'get',
        'keys',
        'schema',
        'defaults',
        'ignore_extra',
        'raise_extra',
        'respect_none',
        '_encode'
    }

    ignore_extra = False
    raise_extra = True
    respect_none = False

    def __new__(
            mcs,
            name: str,
            bases: tuple[type, ...],
            attrs: dict[str, Any],
            **kwargs: Any
    ) -> 'SchemaMeta':
        cls = super().__new__(mcs, name, bases, attrs)

        # Keywords passed on inheriting from an instance of this metaclass.
        cls.ignore_extra = kwargs.pop('ignore_extra', mcs.ignore_extra)
        cls.raise_extra = kwargs.pop('raise_extra', mcs.raise_extra)
        cls.respect_none = kwargs.pop('respect_none', mcs.respect_none)

        # The schema is in the class __annotations__!
        raw_schema = mcs.__get(cls, '__annotations__')
        # Extract (string) keys from additional keyword arguments
        kwarg_schema = {key: str for key in kwargs}
        # Fields defined in the class body overwrite keyword fields
        schema = kwarg_schema | raw_schema
        cls.schema = mcs.__validate_schema(name, schema, mcs.__blacklist__)

        # Default values for class variables are in the class __dict__!
        raw_defaults = mcs.__filter(cls, mcs.__get(cls, '__dict__'))
        # Defaults for additional fields from keyword arguments are the keys
        kwarg_defaults = {key: key for key in kwargs if key not in raw_schema}
        # Fields defined in the class body overwrite keyword fields
        defaults = kwarg_defaults | raw_defaults
        cls.defaults = mcs.__validate_defaults(defaults, cls.schema)

        # JSON fields that conflict with methods or properties are forbidden!
        cls.__blacklist__ = mcs.__blacklist__

        return cls

    @staticmethod
    def __get(cls: 'SchemaMeta', attribute: str) -> dict[str, type]:
        """Accumulate dictionary class variables down the inheritance tree."""
        # Get class ancestors starting with the oldest.
        lineage = reversed(cls.mro())
        # Accumulate inherited dictionary attributes, overwriting old with new.
        return reduce(cls.__merge(attribute), lineage, {})

    @staticmethod
    def __merge(attribute: str) -> Callable[[Json, type], Json]:
        """Provide update function for dictionary class attributes."""

        def update(older: Json, newer: type) -> Json:
            """Update dictionary attribute of parent with that of child."""
            return {**older, **getattr(newer, attribute, {})}

        return update

    @staticmethod
    def __filter(cls: 'SchemaMeta', defaults: Json) -> Json:
        """Filter down class __dict__ to keys present in the schema."""
        return {key: defaults[key] for key in defaults if key in cls.schema}

    @staticmethod
    def __validate_schema(
            name: str,
            schema: dict[str, type],
            blacklist: set[str]
    ) -> dict[str, type]:
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
    def __validate_defaults(defaults: Json, schema: dict[str, type]) -> Json:
        """Validate that class-variable defaults are sane."""
        none_defaults = []
        none_msg = ''
        cast_defaults = []
        cast_msg = ''
        for item in defaults:
            # Check None values.
            default_is_none = defaults[item] is None
            type_is_not_maybe = not isinstance(schema[item], Maybe)
            if default_is_none and type_is_not_maybe:
                none_defaults.append(item)
                none_msg = (f'\nFor defaults {none_defaults} to be None, mark'
                            ' them as Maybe(<YOUR_TYPE>) in the schema!')
            # Check can cast values.
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
