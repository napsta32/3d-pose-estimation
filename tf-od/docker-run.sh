#!/bin/bash

docker build -t napsta32/tf-od .
docker run \
    --gpus all \
    --user $(id -u):$(id -g) \
    --rm -it \
    -p 8090:8090 \
    -v $(pwd):/home/tensorflow/project \
    napsta32/tf-od
