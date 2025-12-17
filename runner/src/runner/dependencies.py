from fastapi import Request

from runner.pipelines.base import Pipeline


def get_pipeline(request: Request) -> Pipeline:
    return request.app.pipeline
