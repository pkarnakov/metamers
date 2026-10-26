# Ideas

Possible next steps, roughly in order.

## Benchmark instead of single runs

Each result in the README is one run on one target.

- A set of 20–50 short captions (COCO-style), several seeds each
  (needed anyway, since GPU kernels are nondeterministic).
- A sweep runner that writes one results table:
  rate of reaching the exact target, step of the first hit, final loss with noise.
- Use it for ablations, e.g. which of the learning rate, noise, and
  regularization makes `cat_perturb_tv` reach the target while `cat_perturb` does not.

## Recognizability by other observers

Whether an image shows the object or only satisfies the optimized model.

- Describe the images with models not used in the optimization:
  Qwen3.5-2B, a VLM of another family.
- Zero-shot CLIP or SigLIP: probability of the target caption against distractor captions.
- Transfer rate to other models as a metric, comparable across parameterizations.
  Relates to Feather et al.: are late-stage metamers recognizable to other observers?
- Would also settle comparisons such as `cat_perturb_tv` against `cat_perturb_tv_reg4`
  (clean line drawing against more detailed shading).

## Transformation robustness and ensembles

- Random shifts, scales, and rotations of the image at every step,
  as in [feature visualization](https://distill.pub/2017/feature-visualization/).
  Complements the noise already used.
- Several noise samples per step.
- An ensemble of models in the loss (e.g. 0.8B and 2B) against exploits specific to one model.

## Learned image priors

- Optimize in the latent space of a pretrained VAE (e.g. from Stable Diffusion).
- A diffusion prior (score distillation).
- Compare with `pixel`, `multigrid`, and `fft` on the same benchmark.

## Engineering

- Move the check of the differentiable preprocessing against the processor into a pytest test.
- CI with `make check` and the tests.
