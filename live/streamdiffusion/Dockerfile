ARG BASE_IMAGE=livepeer/ai-runner:live-base
FROM ${BASE_IMAGE}

RUN apt update && apt install -yqq \
    wget libcairo2-dev libgirepository1.0-dev pkg-config \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/live/streamdiffusion

COPY runner/pyproject.toml runner/uv.lock /app/runner/
COPY live/streamdiffusion/pyproject.toml live/streamdiffusion/uv.lock .
# Copy stubs for editable install validation
COPY runner/src/runner/__init__.py /app/runner/src/runner/
COPY live/streamdiffusion/pipeline/__init__.py ./pipeline/

RUN uv sync --locked --no-install-project

RUN uv run python -m streamdiffusion.tools.install-tensorrt

COPY runner/src/runner/ /app/runner/src/runner/
COPY runner/images/ /app/runner/images/
COPY live/streamdiffusion/pipeline ./pipeline/
COPY live/streamdiffusion/default_params ./default_params/
COPY live/streamdiffusion/main.py ./main.py

# Need --inexact to keep packages installed from install-tensorrt
RUN uv sync --locked --inexact

RUN ln -s /models/StreamDiffusion--engines ./engines \
    && mkdir -p /root && ln -s /models/insightface /root/.insightface \
    && ln -s /models/StreamDiffusion--models ./models

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_OFFLINE=1

ARG GIT_SHA
ARG VERSION="undefined"
# SUBVARIANT controls which models are built during prepare and which default params are loaded.
# Empty (default): builds ALL models, uses sdturbo default params at runtime.
# Specific value (sdturbo, sdxl, etc.): builds only those models, uses matching default params.
ARG SUBVARIANT=""

ENV GIT_SHA="${GIT_SHA}" \
    VERSION="${VERSION}" \
    SUBVARIANT="${SUBVARIANT}"

CMD ["uv", "run", "--frozen", "python", "main.py"]
