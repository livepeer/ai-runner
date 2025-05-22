# Development Guide

This document offers a comprehensive guide for configuring the development environment and debugging the [worker](/worker) component within the [AI worker](https://github.com/livepeer/ai-worker) repository.

## Running with Docker

### Get the Docker Image

Ensure [Docker](https://docs.docker.com/engine/install/ubuntu/) and the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) are installed, with Docker correctly [configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker). Then, either pull the pre-built image from DockerHub or build the image locally in this directory.

#### Pull Docker image

```bash
docker pull livepeer/ai-runner:latest
```

#### Build Docker image

```bash
docker build -t livepeer/ai-runner:latest .
```

### Download AI Models

The AI Runner container's runner app uses [HuggingFace](https://huggingface.co/) model IDs to reference models. It expects model checkpoints in a `/models` directory, which can be mounted to a local `models` directory. To download model checkpoints to the local `models` directory, use the `dl-checkpoints.sh` script:

1. **Install Hugging Face CLI**: Install the Hugging Face CLI by running the following command:

   ```bash
   pip install huggingface_hub[cli,hf_transfer]
   ```

2. **Set up Hugging Face Access Token**: Generate a Hugging Face access token as per the [official guide](https://huggingface.co/docs/hub/en/security-tokens) and assign it to the `HG_TOKEN` environment variable. This token enables downloading of [private models](https://huggingface.co/docs/transformers.js/en/guides/private) from the Hugging Face model hub. Alternatively, use the Hugging Face CLI's [login command](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login) to install the token.

   > [!IMPORTANT]
   > The `dl_checkpoints.sh` script includes the [SVD1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) model. To use this model on the _AI Subnet_, visit its [page](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1), log in, and accept the terms.

3. **Download AI Models**: Use the [dl_checkpoints.sh](https://github.com/livepeer/ai-worker/blob/main/runner/dl_checkpoints.sh) script to download models from `aiModels.json` to `~/.lpData/models`. Run the following command in the `.lpData` directory:

   ```bash
   curl -s https://raw.githubusercontent.com/livepeer/ai-worker/main/runner/dl_checkpoints.sh | bash -s -- --beta
   ```

   > [!NOTE]
   > The `--beta` flag downloads only models supported by the Livepeer.inc Gateway node on the _AI Subnet_. Remove this flag to download all models.

> [!TIP]
> If you want to download individual models you can checkout the `dl-checkpoints.sh` script to see how to download model checkpoints to the local `models` directory.

### Run the Pipelines

#### Run text-to-image container

Run container:

```bah
docker run --name text-to-image -e PIPELINE=text-to-image -e MODEL_ID=<MODEL_ID> --gpus <GPU_IDS> -p 8000:8000 -v ./models:/models livepeer/ai-runner:latest
```

Query API:

```bash
curl -X POST -H "Content-Type: application/json" localhost:8000/text-to-image -d '{"prompt":"a mountain lion"}'
```

Retrieve the image from the response:

```bash
curl -O 0.0.0.0:8937/stream/<STREAM_URL>.png
```

#### Run image-to-image container

Run container:

```bash
docker run --name image-to-image -e PIPELINE=image-to-image -e MODEL_ID=<MODEL_ID> --gpus <GPU_IDS> -p 8000:8000 -v ./models:/models livepeer/ai-runner:latest
```

Query API:

```bash
curl -X POST localhost:8000/image-to-image -F prompt="a mountain lion" -F image=@<IMAGE_FILE>
```

Retrieve the image from the response:

```bash
curl -O 0.0.0.0:8937/stream/<STREAM_URL>.png
```

#### Run image-to-video container

Run container

```bah
docker run --name image-to-video -e PIPELINE=image-to-video -e MODEL_ID=<MODEL_ID> --gpus <GPU_IDS> -p 8000:8000 -v ./models:/models livepeer/ai-runner:latest
```

Query API:

```bash
curl -X POST localhost:8000/image-to-video -F image=@<IMAGE_FILE>
```

Retrieve the image from the response:

```bash
curl -O 0.0.0.0:8937/stream/<STREAM_URL>.mp4
```

### Test with Swagger UI

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-03-swagger-02.png)

The runner comes with a built-in [Swagger UI](https://fastapi.tiangolo.com/features/) for testing API endpoints. Visit `http://localhost:8000/docs` to access it.

### Apply Pipeline Optimizations

The runner contains several optimizations to speed up inference or reduce memory usage:

- `SFAST`: Enables the [stable-fast](https://github.com/chengzeyi/stable-fast) optimization, enhancing inference performance.

### Run benchmarking Script

```bash
docker run --gpus <GPU_IDs> -v ./models:/models livepeer/ai-runner:latest python bench.py --pipeline <PIPELINE> --model_id <MODEL_ID> --runs <RUNS> --batch_size <BATCH_SIZE>
```

Example command:

```bash
# Benchmark the text-to-image pipeline with the stabilityai/sd-turbo model over 3 runs using GPU 0
docker run --gpus 0 -v ./models:/models livepeer/ai-runner:latest python bench.py --pipeline text-to-image --model_id stabilityai/sd-turbo --runs 3
```

Example output:

```bash
----AGGREGATE METRICS----


pipeline load time: 1.473s
pipeline load max GPU memory allocated: 2.421GiB
pipeline load max GPU memory reserved: 2.488GiB
avg inference time: 0.482s
avg inference time per output: 0.482s
avg inference max GPU memory allocated: 3.024s
avg inference max GPU memory reserved: 3.623s
```

For benchmarking script usage information:

```bash
docker run livepeer/ai-runner:latest python bench.py -h
```

### Deploy on Modal

To deploy the runner on [Modal](https://modal.com/), a serverless GPU platform, follow these steps:

1. **Prepare for Deployment**: Use the `modal_app.py` file for deployment. Before you start, perform a dry-run in a dev [environment](https://modal.com/docs/reference/cli/environment) to ensure everything is set up correctly.

2. **Create Secrets**: In the Modal dashboard, create an `api-auth-token` [secret](https://modal.com/docs/guide/secrets#secrets) and assign the `AUTH_TOKEN` variable to it. This secret is necessary for authentication. If you're using gated HuggingFace models, create a `huggingface` secret and set the `HF_TOKEN` variable to your HuggingFace access token.

3. **Create a Volume**: Run `modal volume create models` to create a network [volume](https://modal.com/docs/guide/volumes#volumes). This volume will store the weights of your models.

4. **Download Models**: For each model ID referenced in `modal_app.py`, run `modal run modal_app.py::download_model --model-id <MODEL_ID>` to download the model.

5. **Deploy the Apps**: Finally, deploy the apps with `modal deploy modal_app.py`. After deployment, the web endpoints of your apps will be visible in your Modal dashboard.

## Debugging

### Local Debugging

To debug the `ai-runner` as used from the AI worker logic in `go-livepeer`, use the go scripts in the [cmd/examples](https://github.com/livepeer/ai-worker/tree/examples) folder to test and debug the AI worker. Run these scripts with [Golang](https://go.dev/) or use [Vscode](https://code.visualstudio.com/) with the [golang extension](https://code.visualstudio.com/docs/languages/go) for debugging. Future updates will include tests to enhance the development pipeline.

### Debugging a custom go-livepeer

To run the AI worker examples with a custom [go-livepeer](https://github.com/livepeer/go-livepeer) version, replace the go module reference in the [`examples/go.mod`](../examples/go.mod) file with the path to your local `go-livepeer` repository:

```bash
go mod edit -replace github.com/livepeer/go-livepeer=../path/to/go-livepeer
```

This setup allows you to debug the AI worker package in `go-livepeer` when making changes to the `ai-runner` software.
