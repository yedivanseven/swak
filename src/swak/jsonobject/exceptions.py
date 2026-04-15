class SchemaError(Exception):
    pass


class DefaultsError(Exception):
    pass


class ParseError(Exception):
    pass


class CastError(Exception):
    pass


class ValidationErrors(ExceptionGroup):
    pass
