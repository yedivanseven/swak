from typing import Any
from functools import cached_property
from google.cloud.storage import Client
from ...misc import ArgRepr



class Gcs(ArgRepr):
    """Wraps a Google Cloud Storage (GCS) client for delayed instantiation.

    For more convenient function chaining in functional compositions, instances
    are also callable and simply return a new client when called.

    Parameters
    ----------
    project: str
        The name of the project the client should act on.
    *args
        Additional arguments to be passed to the client init.
    **kwargs
        Additional keyword arguments to be passed to the client init.
        See the `reference <https://cloud.google.com/python/docs/reference/
        storage/latest/google.cloud.storage.client.Client>`_ for options.

    """

    def __init__(self, project: str, *args: Any, **kwargs: Any) -> None:
        self.project = project.strip(' /.')
        self.args = args
        self.kwargs = kwargs
        super().__init__(self.project, *self.args, **self.kwargs)

    @cached_property
    def client(self) -> Client:
        """New GCS client on first request, cached for subsequent requests."""
        return Client(self.project, *self.args, **self.kwargs)

    def __call__(self, *_: Any, **__: Any) -> Client:
        """New GCS client on every call, ignoring any (keyword) arguments."""
        return Client(self.project, *self.args, **self.kwargs)
