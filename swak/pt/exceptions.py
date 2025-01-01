class LossError(Exception):
    pass


class EmbeddingError(Exception):
    pass


class TrainError(Exception):
    pass


class CompileError(Exception):
    pass


class ValidationErrors(ExceptionGroup):
    pass


class ShapeError(Exception):
    pass


class DeviceError(Exception):
    pass


class DTypeError(Exception):
    pass
