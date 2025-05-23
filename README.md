# ai-runner

> [!WARNING]
> The AI network is in it's **Beta** phase and although it is ready for production it is still under development. Please report any issues you encounter to the [Livepeer Discord](https://discord.com/invite/livepeer).

This repository hosts the AI runner for processing AI inference jobs on the Livepeer network.

## Overview

The AI runner is a containerized Python application which processes inference requests on Livepeer AI's Pipelines and models. It loads models into GPU memory and exposes a REST API other programs like [the Livepeer node AI worker](https://github.com/livepeer/ai-runner/blob/main/README.md) can use to request AI inference requests. The AI runner code sits in the [runner](https://github.com/livepeer/ai-runner/tree/main/runner) directory.

## Build

To build the AI runner locally and run examples, follow these steps:

1. Follow the instructions in this document to download model checkpoints and build the runner image.
2. Generate Go bindings for the runner OpenAPI spec with `make codegen`.
3. Run any examples in the `cmd/examples` directory, e.g., `go run cmd/examples/text-to-image/main.go <RUNS> <PROMPT>`.

## Architecture

A high level sketch of how the runner is used:

![Architecture](./docs/images/architecture.png)

The AI runner, found in the [app](./runner/app) directory, consists of:

- **Routes**: FastAPI routes in [app/routes](./runner/app/routes) that handle requests and delegate them to the appropriate pipeline.
- **Pipelines**: Modules in [app/pipelines](./runner/app/pipelines) that manage model loading, request processing, and response generation for specific AI tasks.

It also includes utility scripts:

- **[bench.py](./runner/bench.py)**: Benchmarks the runner's performance.
- **[gen_openapi.py](./runner/gen_openapi.py)**: Generates the OpenAPI specification for the runner's API endpoints.
- **[dl_checkpoints.sh](./runner/dl_checkpoints.sh)**: Downloads model checkpoints from Hugging Face.
- **[modal_app.py](./runner/modal_app.py)**: Deploys the runner on [Modal](https://modal.com/), a serverless GPU platform.

## OpenAPI Specification

Regenerate the OpenAPI specification for the AI runner's API endpoints with:

```bash
python gen_openapi.py
```

To correspondingly generate the Go client bindings in the go-livepeer repository,
you should clone [`livepeer/go-livepeer`](https://github.com/livepeer/go-livepeer) and run:
```bash
# in the go-livepeer repo
make ai_worker_codegen
```

Alternatively, if you want to test the client from a development version of
`ai-runner`, you can specify a commit hash or branch to generate from. e.g.:
```bash
# for commit `aa7ab76`
make ai_worker_codegen REF=aa7ab76
# for branch `vg/test`
make ai_worker_codegen REF=refs/heads/vg/test
```

## Development documentation

For more on developing and debugging the AI runner, see the [development documentation](./docs/development-guide.md).

## Credits

Based off of [this repo](https://github.com/huggingface/api-inference-community/tree/main/docker_images/diffusers).

