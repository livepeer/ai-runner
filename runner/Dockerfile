# Based on https://github.com/huggingface/api-inference-community/blob/main/docker_images/diffusers/Dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Yondon Fu <yondon@livepeer.org>"

# Add any system dependency here
# RUN apt-get update -y && apt-get install libXXX -y

ENV DEBIAN_FRONTEND=noninteractive \
    PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install prerequisites
RUN apt update && \
    apt install -yqq build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git \
    ffmpeg

ARG PYTHON_VERSION=3.11
ARG PIP_VERSION=23.3.2

SHELL [ "/bin/bash", "-c" ]

# Install pyenv
RUN curl https://pyenv.run | bash

RUN source /root/.bashrc && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Upgrade pip and install your desired packages
RUN pip install --no-cache-dir --upgrade pip==${PIP_VERSION} setuptools==69.5.1 wheel==0.43.0 && \
    pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

ARG VERSION="undefined"
ENV VERSION=${VERSION}

WORKDIR /app

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --requirement /tmp/requirements.txt

RUN pip install https://github.com/chengzeyi/stable-fast/releases/download/v1.0.3/stable_fast-1.0.3+torch211cu121-cp311-cp311-manylinux2014_x86_64.whl

# Most DL models are quite large in terms of memory, using workers is a HUGE
# slowdown because of the fork and GIL with python.
# Using multiple pods seems like a better default strategy.
# Feel free to override if it does not make sense for your library.
ARG max_workers=1
ENV MAX_WORKERS=$max_workers \
    HUGGINGFACE_HUB_CACHE=/models \
    DIFFUSERS_CACHE=/models \
    MODEL_DIR=/models

COPY app/ /app/app
COPY images/ /app/images
COPY bench.py /app/bench.py
COPY example_data/ /app/example_data

CMD [ "uvicorn", "app.main:app", "--log-config", "app/cfg/uvicorn_logging_config.json", "--host", "", "--port", "8000" ]
