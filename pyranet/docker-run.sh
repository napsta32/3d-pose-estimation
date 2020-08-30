#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

docker build -t pyranet .
docker run --gpus all --rm -it -v $(pwd):/app -v /home/fpena/datasets/mpii/images:/app/data/mpii/images pyranet
