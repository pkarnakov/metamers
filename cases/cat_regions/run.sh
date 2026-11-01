#!/bin/sh -eu
cd "$(dirname "$0")"
# Each run in its own directory named by the start time.
out=$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
cp run.sh "$out/"
../../metamers.py \
  --target "A cat sitting on a table." \
  --size 512 --grayscale \
  --lr 0.01 --steps 500 --lr_half 200 \
  --shift 64 \
  --reg_tv 0.1 \
  --out "$out"
