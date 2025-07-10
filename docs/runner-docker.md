# Runner Docker Images

This folder contains Dockerfiles for pipelines supported by the Livepeer AI network. The list is maintained by the Livepeer community and audited by the [Core AI team](https://explorer.livepeer.org/treasury/42084921863832634370966409987770520882792921083596034115019946998721416745190). In the future, we will enable custom pipelines to be used with the Livepeer AI network.

## Building a Pipeline-Specific Container

> [!NOTE]
> We are transitioning our existing pipelines to this new structure. As a result, the base container is currently somewhat bloated. In the future, the base image will contain only the necessary dependencies to run any pipeline.

All pipeline-specific containers are built on top of the base container found in the main [runner](../) folder and on [Docker Hub](https://hub.docker.com/r/livepeer/ai-runner). The base container includes the minimum dependencies to run any pipeline, while pipeline-specific containers add the necessary dependencies for their respective pipelines. This structure allows for faster build times, less dependency bloat, and easier maintenance.

### To build a pipeline-specific container (non-ComfyUI)

To build a pipeline-specific container, you need to build the base container first. The base container is tagged as `base`, and the pipeline-specific container is built from the Dockerfile in the pipeline-specific folder. For example, to build the `segment-anything-2` pipeline-specific container, follow these steps:

1. **Navigate to the `ai-worker/runner` Directory**:

   ```bash
   cd ai-worker/runner
    ```

2. **Build the Base Container**:

   ```bash
   docker build --build-arg VERSION=$(./print_version.sh) -t livepeer/ai-runner:base .
   ```

   This command builds the base container and tags it as `livepeer/ai-runner:base`.

3. **Build the `segment-anything-2` Pipeline-Specific Container**:

   ```bash
   docker build -f docker/Dockerfile.segment_anything_2 -t livepeer/ai-runner:segment-anything-2 .
   ```

   This command builds the `segment-anything-2` pipeline-specific container using the Dockerfile located at [docker/Dockerfile.segment_anything_2](docker/Dockerfile.segment_anything_2) and tags it as `livepeer/ai-runner:segment-anything-2`.

### To build the ComfyUI based pipeline

We provide a convenient build script that simplifies the build and run process. The script is located at `runner/build.sh` and supports multiple operations:

```bash
./build.sh [command]
```

Available commands:
- `base` - Build only the base ComfyUI image
- `app` - Build only the application image
- `models` - Download required model checkpoints and build TensorRT engines
- `run` - Run the ComfyUI container
- `all` - Execute all build steps in sequence (recommended for first-time setup)

For a complete first-time setup, simply run:

```bash
cd ai-worker/runner
./build.sh all
```

This will:
1. Build the base ComfyUI image
2. Build the application image
3. Download required model checkpoints
4. Build TensorRT engines

Once built, you can start the container anytime with:

```bash
./build.sh run
```

This will start the container with all necessary configurations, including GPU support, port mapping (8000), and required environment variables.

### To build the no-op pipeline for local testing (works on Darwin as well)

1. Build Docker images
```
export PIPELINE=noop
docker build --build-arg VERSION=$(./print_version.sh) -t livepeer/ai-runner:live-base -f docker/Dockerfile.live-base .
docker build --build-arg VERSION=$(./print_version.sh) -t livepeer/ai-runner:live-app-${PIPELINE} -f docker/Dockerfile.live-app-noop .
```

2. Start Docker container
```
docker run -it --rm --name video-to-video -p 8000:8000 -e PIPELINE=live-video-to-video -e MODEL_ID=noop livepeer/ai-runner:live-app-noop
```
