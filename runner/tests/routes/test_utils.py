import json

import pytest
import torch
from fastapi import status
from fastapi.responses import JSONResponse

from runner.pipelines.utils import LoraLoadingError
from runner.routes.utils import handle_pipeline_exception


class TestHandlePipelineException:
    """Tests for the handle_pipeline_exception function."""

    @staticmethod
    def parse_response(response: JSONResponse):
        """Parses the JSON response body from a FastAPI JSONResponse object."""
        return json.loads(response.body)

    @pytest.mark.parametrize(
        "exception, expected_status, expected_message, description",
        [
            (
                Exception("Unknown error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Pipeline error.",
                "Returns default message and status code for unknown error.",
            ),
            (
                torch.cuda.OutOfMemoryError("Some Message"),
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "GPU out of memory.",
                "Returns global message and status code for type match.",
            ),
            (
                Exception("CUDA out of memory"),
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Out of memory.",
                "Returns global message and status code for pattern match.",
            ),
            (
                LoraLoadingError("A custom error message"),
                status.HTTP_400_BAD_REQUEST,
                "A custom error message.",
                "Forwards exception message if configured with None.",
            ),
            (
                ValueError("A custom error message"),
                status.HTTP_400_BAD_REQUEST,
                "Pipeline error.",
                "Returns default message if configured with ''.",
            ),
        ],
    )
    def test_handle_pipeline_exception(
        self, exception, expected_status, expected_message, description
    ):
        """Test that the handle_pipeline_exception function returns the correct status
        code and error message for different types of exceptions.
        """
        response = handle_pipeline_exception(exception)
        response_body = self.parse_response(response)
        assert response.status_code == expected_status, f"Failed: {description}"
        assert (
            response_body["detail"]["msg"] == expected_message
        ), f"Failed: {description}"

    def test_handle_pipeline_exception_custom_default_message(self):
        """Test that a custom default error message is used when provided."""
        exception = ValueError("Some value error")
        response = handle_pipeline_exception(
            exception, default_error_message="A custom error message."
        )
        response_body = self.parse_response(response)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response_body["detail"]["msg"] == "A custom error message."

    def test_handle_pipeline_exception_custom_status_code(self):
        """Test that a custom default status code is used when provided."""
        exception = Exception("Some value error")
        response = handle_pipeline_exception(
            exception, default_status_code=status.HTTP_404_NOT_FOUND
        )
        response_body = self.parse_response(response)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response_body["detail"]["msg"] == "Pipeline error."

    def test_handle_pipeline_exception_custom_error_config(self):
        """Test that custom error configuration overrides the global error
        configuration, which prints the exception message.
        """
        exception = LoraLoadingError("Some error message.")
        response = handle_pipeline_exception(
            exception,
            custom_error_config={
                "LoraLoadingError": ("Custom message.", status.HTTP_400_BAD_REQUEST)
            },
        )
        response_body = self.parse_response(response)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response_body["detail"]["msg"] == "Custom message."
