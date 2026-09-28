#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py --model Qwen/Qwen3.5-2B \
  --target "A glass of wine on a table." \
  --size 512 --grayscale \
  --lr 0.002 --steps 1000 \
  --noise 0.2 \
  --out .
