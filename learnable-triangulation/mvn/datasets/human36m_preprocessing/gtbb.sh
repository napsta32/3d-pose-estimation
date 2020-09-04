#!/bin/bash

REPO=/home/fpena/tesis/learnable-triangulation

python3 collect-bboxes.py /home/fpena/datasets/h36m 2
# python generate-labels-npy-multiview.py /datasets/h36m ${REPO}/data/human36m/extra/3d-pose-baseline/h36m 
# python3 generate-labels-npy-multiview.py /datasets/h36m 