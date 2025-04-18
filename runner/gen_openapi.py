import argparse
import copy
import json
import logging
import os
import pathlib

import yaml
from fastapi.openapi.utils import get_openapi

from app.main import app
from app.routes import (
    audio_to_text,
    hardware,
    health,
    version,
    image_to_image,
    image_to_text,
    image_to_video,
    live_video_to_video,
    llm,
    segment_anything_2,
    text_to_image,
    text_to_speech,
    upscale,
)

logging.basicConfig(
    # Making the script futureproof, where we generate the spec in workflows
    level=(
        logging.DEBUG
        if any([x in os.environ for x in ("CI", "DEBUG")])
        else logging.INFO
    ),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Specify Endpoints for OpenAPI schema generation.
SERVERS = [
    {
        "url": "https://dream-gateway.livepeer.cloud",
        "description": "Livepeer Cloud Community Gateway",
    },
    {
        "url": "https://livepeer.studio/api/beta/generate",
        "description": "Livepeer Studio Gateway",
    },
]


def translate_to_gateway(openapi: dict) -> dict:
    """Translate the OpenAPI schema from the 'runner' entrypoint to the 'gateway'
    entrypoint created by the https://github.com/livepeer/go-livepeer package.

    .. note::
        Differences between 'runner' and 'gateway' entrypoints:
        - 'model_id' is enforced in all endpoints.
        - 'metadata' property is removed from all schemas.
        - 'VideoResponse' schema is updated to match the Gateway's transcoded mp4
            response.

    Args:
        openapi: The OpenAPI schema to be translated.

    Returns:
        The translated OpenAPI schema.
    """
    # Enforce 'model_id' in all endpoints
    logger.debug("Enforcing 'model_id' in all endpoints...")
    for _, methods in openapi["paths"].items():
        for _, details in methods.items():
            if "requestBody" in details:
                for _, content_details in details["requestBody"]["content"].items():
                    if (
                        "schema" in content_details
                        and "$ref" in content_details["schema"]
                    ):
                        ref = content_details["schema"]["$ref"]
                        schema_name = ref.split("/")[-1]
                        schema = openapi["components"]["schemas"][schema_name]
                        schema.setdefault("required", [])
                        if "model_id" in schema["properties"]:
                            schema["required"].append("model_id")

                        # Remove 'metadata' property if it exists.
                        if "metadata" in schema["properties"]:
                            schema["properties"].pop("metadata")

    # Update the 'VideoResponse' schema to match the Gateway's response.
    # NOTE: This is necessary because the Gateway transcodes the runner's response and
    # returns an mp4 file.
    logger.debug("Updating 'VideoResponse' schema...")
    openapi["components"]["schemas"]["VideoResponse"] = copy.deepcopy(
        openapi["components"]["schemas"]["ImageResponse"]
    )
    openapi["components"]["schemas"]["VideoResponse"]["title"] = "VideoResponse"

    return openapi


def write_openapi(fname: str, entrypoint: str = "runner"):
    """Write OpenAPI schema to file.

    Args:
        fname: The file name to write to. The file extension determines the file
            type. Either 'json' or 'yaml'.
        entrypoint: The entrypoint to generate the OpenAPI schema for, either
            'gateway' or 'runner'. Default is 'runner'.
    """
    if entrypoint != "gateway":
        app.include_router(health.router)
        app.include_router(hardware.router)
        app.include_router(version.router)
    app.include_router(text_to_image.router)
    app.include_router(image_to_image.router)
    app.include_router(image_to_video.router)
    app.include_router(upscale.router)
    app.include_router(audio_to_text.router)
    app.include_router(segment_anything_2.router)
    app.include_router(llm.router)
    app.include_router(image_to_text.router)
    app.include_router(live_video_to_video.router)
    app.include_router(text_to_speech.router)

    logger.info(f"Generating OpenAPI schema for '{entrypoint}' entrypoint...")
    openapi = get_openapi(
        title="Livepeer AI Runner",
        version=os.getenv("VERSION", pathlib.Path("VERSION").open().read().strip()),
        openapi_version=app.openapi_version,
        description="An application to run AI pipelines",
        routes=app.routes,
        servers=SERVERS,
        separate_input_output_schemas=False,
    )

    # Translate OpenAPI schema to 'gateway' side entrypoint if requested.
    if entrypoint == "gateway":
        logger.info(
            "Translating OpenAPI schema from 'runner' to 'gateway' entrypoint..."
        )
        openapi = translate_to_gateway(openapi)
        fname = f"gateway.{fname}"

    # Write OpenAPI schema to file.
    with open(fname, "w") as f:
        logger.info(f"Writing OpenAPI schema to '{fname}'...")
        if fname.endswith(".yaml"):
            f.write("# !!Auto-generated by 'gen_openapi.py'. DO NOT EDIT!!\n")
            yaml.dump(
                openapi,
                f,
                sort_keys=False,
            )
        else:
            json.dump(
                openapi,
                f,
                indent=4,  # Make human readable.
            )
        logger.info("OpenAPI schema generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        choices=["json", "yaml"],
        default="yaml",
        help="File type to write to, either 'json' or 'yaml'. Default is 'yaml'",
    )
    parser.add_argument(
        "--entrypoint",
        type=str,
        choices=["gateway", "runner"],
        default=["gateway", "runner"],
        nargs="+",
        help=(
            "The entrypoint to generate the OpenAPI schema for, options are 'runner' "
            "and 'gateway'. Default is both."
        ),
    )
    args = parser.parse_args()

    # Generate orchestrator and Gateway facing OpenAPI schemas.
    logger.info("Generating OpenAPI schema.")
    entrypoints = sorted(args.entrypoint, key=lambda x: x != "gateway")
    for entrypoint in entrypoints:
        write_openapi(f"openapi.{args.type.lower()}", entrypoint=entrypoint)
