import warnings
import torch as pt
from pathlib import Path
from ..magic import ArgRepr
from ..text import NotFound
from .types import Module, Device


# ToDo: Write unit tests and docstrings!
class StateSaver(ArgRepr):

    def __init__(self, path: str) -> None:
        self.path = str(Path(path.strip()).resolve())
        super().__init__(self.path)

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        file = self.path.format(*parts)
        pt.save(model.state_dict(), file)
        return ()


class StateLoader(ArgRepr):

    def __init__(
            self,
            path: str,
            map_location: Device | str | None = None,
            not_found: str = NotFound.RAISE
    ) -> None:
        self.path = str(Path(path.strip()).resolve())
        self.map_location = map_location
        self.not_found = not_found
        super().__init__(self.path, self.map_location, not_found)

    def __call__(self, model: Module, *parts: str) -> Module:
        file = self.path.format(*parts)
        try:
            update = pt.load(file, self.map_location, weights_only=True)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty TOML.'
                    warnings.warn(msg.format(file))
                    update = {}
                case NotFound.IGNORE:
                    update = {}
                case _:
                    raise error
        _ = model.load_state_dict(model.state_dict() | update)
        return model.to(self.map_location)


class ModelSaver(ArgRepr):

    def __init__(self, path: str) -> None:
        self.path = str(Path(path.strip()).resolve())
        super().__init__(self.path)

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        file = self.path.format(*parts)
        pt.save(model, file)
        return ()


class ModelLoader(ArgRepr):

    def __init__(
            self,
            path: str = '',
            map_location: Device | str | None = None,
    ) -> None:
        self.path = str(Path(path.strip()).resolve())
        self.map_location = map_location
        super().__init__(self.path, self.map_location)

    def __call__(self, path: str = '', *parts: str) -> Module:
        path = '/' + path.strip(' /') if path.strip(' /') else ''
        file = (self.path + path).format(*parts)
        model = pt.load(file, self.map_location, weights_only=True)
        return model.to(self.map_location)
