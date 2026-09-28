#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py \
  --target "A school bus on a road." \
  --size 512 --grayscale \
  --lr 0.005 --steps 1000 \
  --noise_mode multigrid \
  --noise 0.25 --noise_coarse 0.75 \
  --perturb 0.01 --perturb_coarse 0.75 \
  --out .
