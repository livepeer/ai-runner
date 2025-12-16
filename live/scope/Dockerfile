ARG BASE_IMAGE=livepeer/ai-runner:live-base
FROM ${BASE_IMAGE}

RUN apt update && apt install -yqq \
    wget git curl \
    build-essential software-properties-common \
    libcairo2-dev libgirepository1.0-dev pkg-config \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    python3-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/live/scope

COPY runner/pyproject.toml runner/uv.lock /app/runner/
COPY live/scope/pyproject.toml live/scope/uv.lock .
# Copy stubs for editable install validation
COPY runner/app/__init__.py /app/runner/app/
COPY live/scope/pipeline/__init__.py ./pipeline/

RUN uv sync --locked --no-install-project

COPY runner/app/ /app/runner/app/
COPY runner/images/ /app/runner/images/
COPY live/scope/pipeline ./pipeline/
COPY live/scope/main.py ./main.py

RUN uv sync --locked --inexact

# Configure Scope models directory
ENV DAYDREAM_SCOPE_MODELS_DIR="/models/Scope--models"

ENV HF_HUB_OFFLINE=1 \
    MAX_WORKERS=1 \
    HUGGINGFACE_HUB_CACHE=/models \
    DIFFUSERS_CACHE=/models \
    MODEL_DIR=/models

ARG GIT_SHA
ARG VERSION="undefined"

ENV GIT_SHA="${GIT_SHA}" \
    VERSION="${VERSION}"

CMD ["uv", "run", "--frozen", "python", "main.py"]

