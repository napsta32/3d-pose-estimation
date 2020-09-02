#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

docker build -t napsta32/videopose3d .
docker run --gpus all \
        --shm-size 50G \
        -v $(pwd):/app \
        -v /home/fpena/datasets/h36m/archives/extracted:/data/cdf \
        --rm -it napsta32/videopose3d bash # -c "cd data && python prepare_data_h36m.py --from-source-cdf /data/cdf"
