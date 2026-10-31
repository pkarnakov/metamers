#!/bin/sh -eu
cd "$(dirname "$0")"
# Each run in its own directory named by the start time.
out=$(date +%Y%m%d-%H%M%S)
mkdir -p "$out"
cp run.sh "$out/"
../../metamers.py \
  --target "A cat sitting on a table." \
  --region "0.1,0.55,0.9,0.95:Table." \
  --region "0.25,0.1,0.75,0.55:Cat." \
  --size 512 --grayscale \
  --lr 0.005 --steps 1000 --lr_half 200 \
  --shift 64 \
  --out "$out"
