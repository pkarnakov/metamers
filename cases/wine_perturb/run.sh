#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py \
  --target "A glass of wine on a table." \
  --size 512 --grayscale \
  --lr 0.01 --steps 1000 \
  --noise_mode multigrid \
  --noise 0.3 --noise_coarse 0.5 \
  --perturb 0.002 --perturb_coarse 0.5 \
  --out .
