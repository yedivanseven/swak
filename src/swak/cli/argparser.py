import sys
import json
from json import JSONDecodeError
from ast import literal_eval
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Any
from itertools import takewhile, dropwhile, chain
from .exceptions import ArgParseError

type Parsed = tuple[list[str], dict[str, Any]]

USAGE = '%(prog)s [action(s)] [-h]'

DESCRIPTION = 'Refer to the README.md for the available actions!'

EPILOG = """
Additionally, all fields of the program's config can be set via
long-format options. Nested fields can be set by dot-separating
levels, e.g., "--root.level1.level2 value"

{!r}
"""


class ArgParser:
    """Parse the command line for actions and any long-format options.

    Using this command-line argument parser alleviates the need for
    defining any groups or options beforehand. Arguments immediately
    following the program call are interpreted as actions to perform as long
    as they do not start with a hyphen. Starting with the first argument
    that starts with a hyphen, command-line arguments will be interpreted as
    ``--key value`` pairs and this long format is the only one allowed.
    Abbreviated options (``-k value``) right after actions (and before any
    long-format options) are ignored.

    Parameters
    ----------
    default_action: str, optional
        Default action to return if none is found in the command-line
        arguments. Defaults to no action.
    usage: str, optional
        Program usage message.
    description: str, optional
        Program description.
    epilog: str, optional
        Text displayed after the help on command-line options.
    fmt_cls: type, optional
        Option passed on to the underlying ``argparse.ArgumentParser``.
        Defaults to ``argparse.RawTextHelpFormatter``

    """

    def __init__(
            self,
            default_action: str | None = None,
            usage: str = USAGE,
            description: str = DESCRIPTION,
            epilog: str = EPILOG,
            fmt_cls: type = RawTextHelpFormatter
    ) -> None:
        self.default_action = default_action
        self.usage = usage
        self.description = description
        self.epilog = epilog
        self.fmt_cls = fmt_cls
        self.__parse = ArgumentParser(
            usage=usage,
            description=description,
            epilog=epilog,
            formatter_class=fmt_cls
        ).parse_known_args

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}({self.default_action}, ...)'

    def __call__(self, args: list[str] | None = None) -> Parsed:
        """Parse the command-line arguments into actions and options.

        Parameters
        ----------
        args: list of str, optional
            The command-line arguments to parse. Mainly a debugging feature.
            If none is given, ``sys.argv[1:]`` will be parsed.

        Returns
        -------
        actions: list of str
            A list with the actions (as strings) to perform. If none are
            found on the command line and no `default_action` is specified,
            that list will be empty.
        options: dict
            Dictionary with keys and values parsed from long-format
            command line arguments.

        """
        args = sys.argv[1:] if args is None else args
        _, unknowns = self.__parse(args)
        actions, args = self.__split(unknowns)
        actions = self.__valid(actions)
        args = self.__compatible(args)
        options = self.__mapped(args)
        return actions, options

    def __split(self, args: list[str]) -> tuple[list[str], list[str]]:
        """Split command-line options into actions and config settings."""
        # Everything before the first option (starting with "-") are actions.
        actions = tuple(takewhile(lambda arg: not arg.startswith('-'), args))
        # If there are none, fall back onto the default action (if specified).
        if not actions and self.default_action is not None:
            actions = self.default_action,
        # Either way, replace dashes with underscores
        actions = (action.lower().replace('-', '_') for action in actions)

        # Everything after the action(s) are options (starting with "--").
        args = dropwhile(lambda arg: not arg.startswith('--'), args)
        # In case options are given as "--key=value", we split.
        args = chain.from_iterable(arg.split('=') for arg in args)
        return list(actions), list(args)

    @staticmethod
    def __valid(actions: list[str]) -> list[str]:
        """Raise if any action string is not a valid python identifier."""
        for action in actions:
            if not action.isidentifier():
                msg = 'Actions must be valid identifiers, unlike "{}"!'
                raise ArgParseError(msg.format(action))
        return actions

    @staticmethod
    def __compatible(args: list[str]) -> list[str]:
        """Raise if a command-line option is not in long format with value."""
        long_form = all(arg.startswith('--') for arg in args[::2])
        alternating = all(not arg.startswith('-') for arg in args[1::2])
        even_number = len(args) % 2 == 0
        if not (long_form and alternating and even_number):
            msg = ('Command-line arguments must be passed in long format (i.e.'
                   ', as "--key value"), and a value must always be present!')
            raise ArgParseError(msg)
        return args

    def __mapped(self, args: list[str]) -> dict[str, Any]:
        """Create dictionary wih config keys and (string) values from args."""
        zipped = list(zip(args[::2], args[1::2]))
        # Drop the first two characters (always "--") and replace "-" with "_".
        return {k[2:].replace('-', '_'): self.__parsed(v) for k, v in zipped}

    @staticmethod
    def __parsed(value: str) -> Any:
        """Try to parse (string) command-line options into python objects."""
        try:
            parsed = json.loads(value)
        except (TypeError, JSONDecodeError):
            try:
                parsed = literal_eval(value)
            except (TypeError, ValueError, SyntaxError):
                parsed = value
        return parsed
