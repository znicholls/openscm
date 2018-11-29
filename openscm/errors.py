class OutOfTimeBoundsError(IndexError):
    """Error raised when the user attempts to step a model beyond its time axis.
    """

    pass


class OverwriteError(AssertionError):
    """Error raised when the user's action will overwrite existing data
    """
    pass
