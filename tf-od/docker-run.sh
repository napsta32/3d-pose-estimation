#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset


docker build -t napsta32/tf-od .
docker run \
    --gpus all \
    --rm -dit \
    -v $(pwd):/root/project \
    -v /home/fpena/datasets/h36m/processed:/root/data/images \
    napsta32/tf-od

## Jupyter command:
# docker run \
#     --gpus all \
#     --rm -it \
#     -p 8090:8090 \
#     -v $(pwd):/root/project \
#     -v /home/fpena/datasets/h36m/processed:/root/data/images \
#     napsta32/tf-od
