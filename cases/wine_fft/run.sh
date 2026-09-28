#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py --method fft --fft_decay 1.25 \
  --target "A glass of wine on a table." \
  --size 512 --grayscale \
  --lr 0.0003 --steps 1000 \
  --noise 0.2 \
  --out .
