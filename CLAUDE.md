# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`metamers.py` optimizes an image so that a Qwen3.5 VLM generates a prescribed `--target` text
(teacher-forced cross-entropy on target tokens). `viewer.py` is a FastAPI web viewer of the runs in
`cases/`. The README documents method and results; keep it in sync when changing defaults or adding
cases.

- Preprocessing (resize, normalization, patching) is reimplemented differentiably and checked
  against the HF processor at startup; success is judged by greedy generation from the 8-bit
  quantized image through the standard processor.
- Perturbation parameterization `--method pixel|multigrid|fft` (multigrid from mODIL, `--mg_loc cell|node`,
  matching `odil.core.interp_to_finer()`), bounded via a sine, with optional noise in the loss.
- `make download` fetches the model once; the script runs offline afterwards.

## Environment and commands

```bash
uv sync        # dependencies in .venv (torch, transformers, flash-linear-attention, fastapi, ...)
make check     # ruff lint + format check (read-only)
make fix       # ruff check --fix + ruff format
```

Ruff: line length 120, double quotes, rules E/W/F/I (E501, E402, E741 ignored).
There is no test suite.

## Pages

`pages/` is gitignored and is a separate clone of the `gh-pages` branch (`make pages`), the report
<https://pkarnakov.github.io/metamers/>, built from `index.md` with pandoc (`make` in `pages/`, KaTeX
for math). Images (`*.png`, `*.jpg`) and logs are gitignored on `main`; do not commit binaries here.
Figures are published by copying into `pages/media/` and committing in `pages/`, and the README
links them by absolute URL `https://pkarnakov.github.io/metamers/media/...`.

## Cases

- Each case is `cases/<name>/run.sh`, a single `metamers.py` invocation with `--out .`, run by
  `make <name>`. New experiments are added as new case directories.
- Only `run.sh` is tracked (`cases/.gitignore`); outputs are written next to it and overwritten
  by a rerun, except images of later steps, which remain from the previous run.
- Never rewrite or truncate the files of a case while its run is live; the process keeps writing at
  its offset and leaves a gap of NUL bytes. A run is live if `train.jsonl` has no done record and
  changed recently (the viewer's status).

## Outputs, shared by metamers.py and viewer.py

Changing one side requires changing the other.

- `config.yaml`: all arguments (`yaml.safe_dump`, enums as strings), written at startup before the
  model is loaded.
- `train.log`: human-readable lines `step= ... output='...'`, also printed to stdout.
- `train.jsonl`: `{"event": "start", "run": "<start time>"}`, then one record per check
  (`step`, `loss`, `reg`, `tokens_ok`, `linf`, `time`, `ms_step`, `output`, `image`), then
  `{"event": "done"}` at the end. The viewer reads it incrementally, detects a rerun by a change of
  the first line, and reports a run without the done record as running or stopped by the log's age.
- `metamer_<step:05d>.png` per check, `metamer.png` for the latest image that produces the target.

## Viewer

- `make viewer` (`./viewer.py --port 8000`); HTML, CSS and JS are in `static/`, the chart uses uPlot
  vendored in `static/vendor/uplot/` so that the viewer works offline.
- The page polls `/api/runs` every 2 s and updates elements in place (images are swapped once
  loaded, the log table gets new rows prepended, the chart gets `setData()` and keeps its zoom), so nothing flickers.
  Keep new UI parts updating locally rather than re-rendering.
- Images are cached for a day under a URL with the run id, and only images named in the current
  run's records are listed.

## Method notes

- fft: `u = irfft2(û · f^(-d))` with `d = --fft_decay`; the gain of the lowest frequency is
  `512^d`, so the learning rate must shrink as `d` grows.
- `--noise` is the standard deviation per multigrid level for `--noise_mode multigrid`, with
  `--noise_coarse q` multiplying level `l` (0 finest) by `q^l`; no normalization.
- `--perturb` adds noise to the parameters after each Adam step; `--perturb_half` and `--noise_half`
  halve the magnitude every H steps, like `--lr_half`, and nothing else.
- Qwen3.5's linear attention (gated delta rule) uses the Triton kernels of `flash-linear-attention`
  if installed, else a slow PyTorch reference; the fallback warning is hidden because
  `metamers.py` silences `transformers` logging. `causal_conv1d` is not installed, its fallback is a
  single `F.conv1d`.
