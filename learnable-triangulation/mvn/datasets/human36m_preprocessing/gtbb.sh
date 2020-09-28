#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

REPO=/home/fpena/tesis/learnable-triangulation

# python3 -u collect-bboxes.py /home/fpena/datasets/h36m 30 >> out.log 2>> err.log
# python3 -u generate-labels-npy-multiview.py \
#         /home/fpena/datasets/h36m \
#         ${REPO}/data/human36m/extra/3d-pose-baseline/h36m \
#         /home/fpena/datasets/h36m/extra/bboxes-Human36M-GT.npy
python3 -u undistort-h36m.py \
        /home/fpena/datasets/h36m \
        /home/fpena/datasets/h36m/extra/human36m-multiview-labels-GTbboxes.npy \
        30

# python generate-labels-npy-multiview.py /datasets/h36m ${REPO}/data/human36m/extra/3d-pose-baseline/h36m 
# python3 generate-labels-npy-multiview.py /datasets/h36m 