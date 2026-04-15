import string
from typing import Any
from collections.abc import Mapping

type M = Mapping[str, Any] | None


class _Template(string.Template):
    """String template accepting also dots in the substitution keys.

    This is a drop-in replacement for the python standard library's
    ``string.Template`` with the only difference that, for braced substitution
    keys (e.g., "My name is ${person.name}.") dots are allowed as well.

    """

    braceidpattern = '(?a:[_a-z][_a-z0-9]*(\\.[_a-z][_a-z0-9]*)*)'


class TemplateRenderer:
    """Substitute bash-style "${key}" placeholders in a template string.

    This is a light wrapper around the standard library's ``string.Template``,
    calling its method ``safe_substitute`` upon instantiation and its method
    ``substitute`` when calling the (callable) instance. At first, not all keys
    need to be provided, but when the object is called and the substitution
    is finalized, all keys must have been provided.

    Parameters
    ----------
    template: str
        String with bash-style "${key}" placeholders to substitute.
    mapping: dict, optional
        Dictionary-like mapping with the placeholders to substitute as keys
        and the values to substitute as values. Defaults to ``None``.
    **kwargs
        Keyword arguments replace respective placeholders with whatever value
        is passed, overwriting values in `mapping`.

    Note
    ----
    Instances evaluate to ``True`` if all keys were provided at instantiation
    and to ``False`` if there are still keys missing.

    """

    def __init__(
            self,
            template: str,
            mapping: M = None,
            **kwargs: Any
    ) -> None:
        mapping = {} if mapping is None else mapping
        # We could also delay this until instances are called
        self.template = _Template(template).safe_substitute(mapping, **kwargs)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        # Since the string template can be very large, we only
        # display up to 30 characters of its first line.
        head = self.template.splitlines()[0].strip()
        suffix = ' ...' if len(head) > 30 else ''
        return f"{cls}('{head[:30].strip() + suffix}')"

    def __str__(self) -> str:
        return self.template

    def __bool__(self) -> bool:
        return not bool(self.identifiers)

    @property
    def identifiers(self) -> list[str]:
        """List of identifiers that still need to be interpolated."""
        return _Template(self.template).get_identifiers()

    def __call__(self, mapping: M = None, **kwargs: Any) -> str:
        """Substitute bash-style "${key}" placeholders in the wrapped template.

        Parameters
        ----------
        mapping: dict, optional
            Dictionary-like mapping with the placeholders to substitute as keys
            and the values to substitute as values. Defaults to ``None``.
        **kwargs
            Keyword arguments replace respective placeholders with whatever
            value is passed, overwriting values in `mapping`.

        Returns
        -------
        str
            String with all placeholders substituted.

        Raises
        ------
        KeyError
            When a value for one or more placeholders is still missing.
        ValueError
            When a key is not a valid python identifier.

        """
        mapping = {} if mapping is None else mapping
        return _Template(self.template).substitute(mapping, **kwargs)


class FormFiller:
    """Substitute bash-style "${key}" placeholders in a template string.

    This is a light wrapper around the standard library's ``string.Template``,
    first caching a dictionary and optional keywords on instantiation, and
    then calling its method ``substitute`` with these cached values when
    calling the (callable) object with the template string.

    Parameters
    ----------
    mapping: dict, optional
        Dictionary-like mapping with the placeholders to substitute as keys
        and the values to substitute as values. Defaults to ``None``.
    **kwargs
        Keyword arguments replace respective placeholders with whatever value
        is passed, overwriting respective values in `mapping`.

    Note
    ----
    Instances evaluate to ``False`` if no keys were provided at instantiation
    and to ``True`` if there are any keys cached.

    """

    def __init__(self, mapping: M = None, **kwargs: Any) -> None:
        self.mapping = kwargs if mapping is None else {**mapping, **kwargs}

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        # Since the mapping can be very large, we only
        # display up to its first 30 characters.
        head = str(self.mapping).splitlines()[0].strip()
        suffix = ' ...}' if len(head) > 30 else ''
        return f'{cls}({head[:30] + suffix})'

    def __bool__(self) -> bool:
        return bool(self.mapping)

    def __call__(self, template: str, mapping: M = None, **kwargs: Any) -> str:
        """Substitute bash-style "${key}" placeholders in a string template.

        Parameters
        ----------
        template: str
            String with bash-style "${key}" placeholders to substitute.
        mapping: dict, optional
            Dictionary-like mapping with the placeholders to substitute as keys
            and the values to substitute as values. Defaults to ``None``.
        **kwargs
            Keyword arguments replace respective placeholders with whatever
            value is passed, overwriting the cached `mapping`.

        Returns
        -------
        str
            String with all placeholders substituted.

        Raises
        ------
        KeyError
            When a value for one or more placeholders is still missing.
        ValueError
            When a key is not a valid python identifier.

        """
        mapping = kwargs if mapping is None else {**mapping, **kwargs}
        return _Template(template).substitute(self.mapping, **mapping)
