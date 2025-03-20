#!/bin/bash
set -ex

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_room> <output_room>"
    exit 1
fi

INPUT_ROOM=$1
OUTPUT_ROOM=$2
PIPELINE=${3:-streamdiffusion}
PORT=${4:-9000}

# Build images, this will be quick if everything is cached
docker build -t livepeer/ai-runner:live-base -f docker/Dockerfile.live-base .
if [ "${PIPELINE}" = "noop" ]; then
    docker build -t livepeer/ai-runner:live-app-noop -f docker/Dockerfile.live-app-noop .
else
    docker build -t livepeer/ai-runner:live-base-${PIPELINE} -f docker/Dockerfile.live-base-${PIPELINE} .
    docker build -t livepeer/ai-runner:live-app-${PIPELINE} -f docker/Dockerfile.live-app__PIPELINE__ --build-arg PIPELINE=${PIPELINE} .
fi

CONTAINER_NAME=live-video-to-video-${PIPELINE}
docker run -it --rm --name ${CONTAINER_NAME} \
  -e PIPELINE=live-video-to-video \
  -e MODEL_ID=${PIPELINE} \
  --gpus all \
  -p ${PORT}:8000 \
  -v ./models:/models \
  livepeer/ai-runner:live-app-${PIPELINE} 2>&1 | tee ./run-lv2v.log &
DOCKER_PID=$!

# make sure to kill the container when the script exits
trap 'docker rm -f ${CONTAINER_NAME}' EXIT

set +x

echo "Waiting for server to start..."
while ! grep -aq "Uvicorn running" ./run-lv2v.log; do
  sleep 1
done
sleep 2
echo "Starting pipeline from ${INPUT_ROOM} to ${OUTPUT_ROOM}..."

set -x

curl -vvv http://localhost:${PORT}/live-video-to-video/ \
  -H 'Content-Type: application/json' \
  -d "{\"publish_url\":\"https://wwgcyxykwg9dys.transfix.ai/trickle/${OUTPUT_ROOM}\",\"subscribe_url\":\"https://wwgcyxykwg9dys.transfix.ai/trickle/${INPUT_ROOM}\"}"

# let docker container take over
wait $DOCKER_PID
