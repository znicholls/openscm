class OutOfBoundsError(IndexError):
    """Error raised when the user attempts to step a model beyond its input data range.
    """

    pass


class OverwriteError(AssertionError):
    """Error raised when the user's action will overwrite existing data
    """

    pass
