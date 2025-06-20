"""Custom exceptions used throughout the whole application."""


class InferenceError(Exception):
    """Exception raised for errors during model inference."""

    def __init__(self, message="Error during model execution", original_exception=None):
        """Initialize the exception.

        Args:
            message: The error message.
            original_exception: The original exception that caused the error.
        """
        if original_exception:
            message = f"{message}: {original_exception}"
        super().__init__(message)
        self.original_exception = original_exception


class InvalidInputError(InferenceError):
    """Exception raised when input validation fails."""
    pass


class ModelOOMError(InferenceError):
    """Exception raised when the model runs out of memory."""
    pass


class GenerationError(InferenceError):
    """Exception raised for general errors in the generation process."""
    pass
