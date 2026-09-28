#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py \
  --target "A glass of wine on a table." \
  --size 512 --grayscale --method pixel \
  --lr 0.03 --steps 400 \
  --noise 0.2 --reg_lap 10000 \
  --out .
