#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py \
  --target "A cat sitting on a table." \
  --size 512 --grayscale \
  --lr 0.005 --steps 1000 \
  --noise_mode multigrid \
  --noise 0.2 --noise_coarse 0.5 \
  --perturb 0.002 --perturb_coarse 0.5 \
  --reg_tv 0 --reg_lap 0.04 --reg_phi 1 \
  --out .
