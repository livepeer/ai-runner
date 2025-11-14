import pathlib
import json
from typing import Union
from pydantic import field_validator

from ..interface import BaseParams

_default_workflow_path = pathlib.Path(__file__).parent.absolute() / "default_workflow.json"
with open(_default_workflow_path, 'r') as f:
    DEFAULT_WORKFLOW_JSON = json.load(f)


class ComfyUIParams(BaseParams):
    class Config:
        extra = "forbid"

    prompt: Union[str, dict] = DEFAULT_WORKFLOW_JSON

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v) -> dict:
        if v == "":
            return DEFAULT_WORKFLOW_JSON

        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed prompt JSON must be a dictionary/object")
                return parsed
            except json.JSONDecodeError:
                raise ValueError("Provided prompt string must be valid JSON")

        raise ValueError(
            "Prompt must be either a JSON object or such JSON object serialized as a string"
        )
