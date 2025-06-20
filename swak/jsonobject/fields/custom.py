from abc import ABC, abstractmethod


# ToDo: Add polars support
class CustomField(ABC):
    """Abstract base class for defining custom field types in a JsonObject."""

    @abstractmethod
    def __init__(self, arg) -> None:
        """Will be called with exactly one argument as a custom field type."""
        ...

    @property
    @abstractmethod
    def as_json(self):
        """How should instances be represented in a JSON-serializable dict?"""
        ...

    @property
    @abstractmethod
    def as_dtype(self):
        """How should instances appear in cells of a pandas Series?"""
        ...
