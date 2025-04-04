#!/bin/bash
set -ex

PIPELINE=noop
PORT=8900

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
sleep 5

set -x

curl --location 'http://127.0.0.1:8900/live-video-to-video/' \
--header 'Content-Type: application/json' \
--data '{
  "subscribe_url": "http://172.17.0.1:3389/sample",
  "publish_url": "http://172.17.0.1:3389/sample-out"
}'

# let docker container take over
wait $DOCKER_PID
