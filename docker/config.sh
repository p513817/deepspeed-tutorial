#!/bin/bash

USER=""
REPO="inno-ds"
TAG="b0.0.1"

if [ -z "$REPO" ]; then
    echo "Error: REPO cannot be empty."
    exit 1
fi

if [ -z "$TAG" ]; then
    echo "Error: TAG cannot be empty."
    exit 1
fi

if [ -z "$USER" ]; then
    DOCKER_IMAGE="$REPO:$TAG"
else
    DOCKER_IMAGE="$USER/$REPO:$TAG"
fi

echo "Docker Image Name: $DOCKER_IMAGE"

# Use the value of the CNTR environment variable if set, otherwise default to REPO
CNTR=${CNTR:-$REPO}
echo "Docker Container Name: ${CNTR}"
