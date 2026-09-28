#!/bin/sh -eu
cd "$(dirname "$0")"
../../metamers.py \
  --target "A black and white photograph of a fluffy cat sitting on a windowsill, looking straight at the camera with large round eyes, pointed ears, and long white whiskers." \
  --size 512 --grayscale \
  --lr 0.002 --steps 1000 \
  --noise 0.2 \
  --out .
