#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

docker build -t napsta32/mspn .
docker run --gpus all --shm-size 50G \
    -v $(pwd):/app \
    -v /home/fpena/datasets/h36m/extra:/data/H36M/bb \
    -v /home/fpena/datasets/h36m/processed:/data/H36M/images \
    -v /home/fpena/datasets/coco:/data/COCO/images \
    --rm -it napsta32/mspn bash
