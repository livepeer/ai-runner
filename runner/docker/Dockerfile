# Based on https://github.com/huggingface/api-inference-community/blob/main/docker_images/diffusers/Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Yondon Fu <yondon@livepeer.org>"

ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt update && \
    apt install -yqq build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
    ffmpeg cuda-toolkit-12-8

ARG PYTHON_VERSION=3.11
ARG PIP_VERSION=23.3.2

# TODO: Remove pyenv section once we're confident on the nexus migration
# --- pyenv for backwards compatibility ---
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
SHELL [ "/bin/bash", "-c" ]
RUN curl https://pyenv.run | bash
RUN source /root/.bashrc && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash
# --- end pyenv ---

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

WORKDIR /app
RUN uv python install $PYTHON_VERSION && uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Install torch (older version for batch pipelines)
RUN uv pip install --no-cache --upgrade pip==${PIP_VERSION} setuptools==69.5.1 wheel==0.43.0 && \
    uv pip install --no-cache torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

ARG VERSION="undefined"
ENV VERSION=${VERSION}

# Copy lockfile and pyproject.toml first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install batch dependencies (--inexact preserves torch installed above)
RUN uv sync --locked --extra batch --no-install-project --inexact

# Install stable-fast
RUN uv pip install --no-cache https://github.com/chengzeyi/stable-fast/releases/download/v1.0.3/stable_fast-1.0.3+torch211cu121-cp311-cp311-manylinux2014_x86_64.whl

# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers \
    HUGGINGFACE_HUB_CACHE=/models \
    DIFFUSERS_CACHE=/models \
    MODEL_DIR=/models

COPY src/runner/ /app/src/runner
COPY images/ /app/images
COPY bench.py /app/bench.py
COPY example_data/ /app/example_data

# Final sync to install the project
RUN uv sync --locked --extra batch --inexact

CMD ["uv", "run", "--frozen", "uvicorn", "runner.main:app", "--log-config", "src/runner/cfg/uvicorn_logging_config.json", "--host", "", "--port", "8000"]
