class AgentError(RuntimeError):
    """Exception raised for errors when calling the various Agent methods at wrong times.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AgentWarning(RuntimeWarning):
    """Warning issued of the various Agent methods.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """


class SliceError(LookupError):
    """Exception raised for errors when getting slices from CircularBuffers.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
