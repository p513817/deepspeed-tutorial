#!/bin/bash

DOCKER_DIR=$(cd $(dirname $0) && pwd)
source "${DOCKER_DIR}/config.sh"

HOST_DIR=$(dirname ${DOCKER_DIR})
cd ${HOST_DIR}
echo "Workspace Path: $(pwd)"

DST_DIR="/opt/inno"
DEF_SECRET="./config/secret.json"
SRC_SECRET=$(realpath ${SECRET:-$DEF_SECRET})
DST_SECRET=${SRC_SECRET/$HOST_DIR/$DST_DIR}

# Check if the container exists
if [ "$(docker ps -a -q -f name=${CNTR})" ]; then
    echo "Container ${CNTR} already exists. Starting and attaching to it."
    docker start ${CNTR}
    docker exec -it ${CNTR} bash
else
    echo "Container ${CNTR} does not exist. Creating and running a new container."
    docker run \
    -dt \
    --restart always \
    --gpus all \
    --name ${CNTR} \
    --network=host \
    --ipc=host \
    --privileged \
    -v $(pwd):${DST_DIR} \
    -v /mnt/nvme2t:${DST_DIR}/nvme2t \
    -e INNO_DS_SECRET=$DST_SECRET \
    -e HF_HOME="${DST_DIR}/cache/huggingface" \
    -e WANDB_DIR="${DST_DIR}/cache" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/timezone:/etc/timezone:ro \
    ${DOCKER_IMAGE}

    docker exec -it ${CNTR} bash
fi
