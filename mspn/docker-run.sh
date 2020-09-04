#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

docker build -t napsta32/mspn .
docker run --gpus all --shm-size 50G -v $(pwd):/app -v /home/fpena/datasets/coco:/app/dataset/COCO/images --rm -it napsta32/mspn bash
