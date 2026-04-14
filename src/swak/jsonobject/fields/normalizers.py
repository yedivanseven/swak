from ...misc import ArgRepr


class Lower(ArgRepr):
    """Lowercase strings, optionally stripping characters left and/or right.

    Parameters
    ----------
    lstrip: str, optional
        Characters to strip from the left of a string. Defaults to ``None``,
        resulting in only whitespaces to be stripped.
    rstrip: str, optional
        Characters to strip from the right of a string. Defaults to ``None``,
        resulting in only whitespaces to be stripped.

    """

    def __init__(
            self,
            lstrip: str | None = None,
            rstrip: str | None = None
    ) -> None:
        super().__init__(lstrip, rstrip)
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __call__(self, arg: str) -> str:
        """Lowercase strings, optionally stripping left and/or right.

        Parameters
        ----------
        arg: str
            String to put into lowercase and optionally strip characters from.

        Returns
        -------
        str
            The lowercased and optionally stripped string.

        """
        return str(arg).lower().lstrip(self.lstrip).rstrip(self.rstrip)


class Strip(ArgRepr):
    """Strip characters from strings on the left and/or right.

    Parameters
    ----------
    left: str, optional
        Characters to strip from the left of a string. Defaults to ``None``,
        resulting in only whitespaces to be stripped.
    right: str, optional
        Characters to strip from the right of a string. Defaults to ``None``,
        resulting in only whitespaces to be stripped.

    """

    def __init__(
            self,
            left: str | None = None,
            right: str | None = None
    ) -> None:
        super().__init__(left, right)
        self.left = left
        self.right = right

    def __call__(self, arg: str) -> str:
        """Strip characters from strings on the left and/or right.

        Parameters
        ----------
        arg: str
            String to strip characters from left and/or right.

        Returns
        -------
        str
            The left/right stripped string.

        """
        return str(arg).lstrip(self.left).rstrip(self.right)
