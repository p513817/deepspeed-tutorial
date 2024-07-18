#!/bin/bash

DOCKER_DIR=$(cd $(dirname $0) && pwd)
cd ${DOCKER_DIR}
source "${DOCKER_DIR}/config.sh"

docker build \
-t $DOCKER_IMAGE \
${DOCKER_DIR}