# Metamers of a vision-language model

Generates images from a description by inverting a vision-language model.
The image is optimized from uniform gray until
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B),
asked to "Describe this image in one sentence.",
answers with exactly the prescribed text,
using Adam and gradients with respect to the image.
With a multigrid parameterization of the image,
noise during the optimization, and random perturbations of the parameters,
the images show the described objects
instead of the adversarial noise that gradient descent on plain pixels produces.

[<img src="https://pkarnakov.github.io/metamers/media/perturb.png" width="800">](https://pkarnakov.github.io/metamers/media/perturb.png)

Images for the targets
"A glass of wine on a table.",
"A school bus on a road.",
and "A cat sitting on a table." (the two right images, the rightmost with a smoothness penalty),
512×512 grayscale, 1000 steps each.
The first and the last give exactly the target,
see [perturbation of the parameters](#perturbation-of-the-parameters).
A shorter report with the best examples is at <https://pkarnakov.github.io/metamers/>.

The term follows the model metamers of
[Feather et al. (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/ac27b77292582bc293a51055bfc994ee-Abstract.html),
stimuli that produce the same activations in a model stage as a natural stimulus,
found by gradient descent from noise.
They found that metamers from late stages of vision and audio models
are often unrecognizable to humans,
see also the follow-up by [Feather et al. (2023)](https://doi.org/10.1038/s41593-023-01442-0).
Here, the matched response is the generated text,
which is the latest possible stage.

```bash
uv sync         # dependencies in .venv, then `source .venv/bin/activate` or prefix with `uv run`
make download   # fetch the model, the script itself runs offline
make help       # list targets
make wine       # run cases/wine/run.sh, output next to the script
./metamers.py --target "A cat." --size 512 --grayscale --noise 0.2 --lr 0.002 --steps 1000 --out out_cat
make viewer     # web viewer of the runs in cases/ at http://127.0.0.1:8000
```

## Method

### Loss

The prompt (default "Describe this image in one sentence.") and the image
are followed by the target tokens $y_1, \dots, y_n$ and the end-of-turn token.
The loss is the teacher-forced cross-entropy
from a single forward pass,

$$L(x) = -\frac{1}{n} \sum_{i=1}^{n} \log p(y_i \mid x, \text{prompt}, y_1, \dots, y_{i-1}).$$

The arguments are saved to `config.yaml` in the output directory, the checks to `train.log` and,
as JSON lines for the viewer, to `train.jsonl` (a start record with the id of the run, one record
per check, and a done record at the end).
The log `train.log` shows `tokens_ok`,
the fraction of target tokens that are the most likely ones.
The target is reached if greedy generation from the image quantized to 8 bits,
passed through the standard processor, gives exactly the target.
The preprocessing (resize, normalization, patches) is reimplemented differentiably
and checked against the processor at startup.
Images below 256×256 are resized to 256×256 by the processor
with bicubic interpolation as in PIL.

### Parameterization

The image is $x = x_0 + u$ with initial image $x_0$ (`--init`, default gray)
and perturbation $u$ represented in one of the ways (`--method`):

* `pixel`: plain pixels,
* `multigrid` (default): the multigrid decomposition from
  [mODIL](https://doi.org/10.1140/epje/s10189-023-00313-7),
  a sum of components on grids coarsened twice per level,
  $u = u_1 + T_1 (u_2 + T_2 (u_3 + \dots))$,
  where $T_i$ is linear interpolation to the finer grid.
  For 512×512, there are 9 levels from 512×512 to 2×2.
* `fft`: the Fourier spectrum scaled by the inverse frequency,
  $u = \mathcal{F}^{-1}(\hat{u} / f^{d})$,
  where $\mathcal{F}$ is the orthonormal 2D real FFT,
  $f$ is the frequency in cycles per pixel, bounded below by the lowest nonzero frequency,
  and $d$ is the decay (`--fft_decay`, default 1),
  as in [feature visualization](https://distill.pub/2017/feature-visualization/) of Olah et al.
  With $d = 0$, the spectrum is equivalent to pixels up to a rotation,
  and with $d = 1$, the zero frequency changes the image as the coarsest multigrid level.

The values of the components are located in cells or nodes (`--mg_loc`):

* `cell` (default): 512, 256, ..., 2 cells coinciding with pixels,
  interpolation with linear extrapolation at the boundaries,
* `node`: 513, 257, ..., 3 nodes, cropped to 512 pixels after interpolation.

Both match `odil.core.interp_to_finer()`,
except for corner cells where ODIL extrapolates along the diagonal.
The components are optimized with Adam
with a constant learning rate by default
(`--lr`, default 0.03 for `multigrid`, 0.01 for `pixel`, and 0.003 for `fft`).
Adam is not invariant to rotations, and its steps on all Fourier coefficients add up in pixels,
so with `fft` the first step at 256×256 changes the image as much as `multigrid`
with a 10 times larger learning rate, and more for larger images.

### Bounds

The image is kept within $[\ell, h]$, which is $[0, 1]$
or $x_0 \pm \varepsilon$ intersected with $[0, 1]$ (`--eps`), through a sine,

$$x = c + r \sin(\varphi), \quad \varphi = \varphi_0 + u / r,$$

with center $c = (\ell + h) / 2$, radius $r = (h - \ell) / 2$,
and $\varphi_0$ such that $u = 0$ gives $x_0$.
Near the center, the image changes by $u$.
Unlike clamping or a sigmoid, the sine has no flat regions:
a value pushed past a bound turns back.

### Noise and regularization

With `--noise σ`, the cross-entropy is evaluated on $x + \sigma \xi$
with Gaussian noise $\xi$, new at every step,
and $\sigma$ halving over `--noise_half` steps if given.
The image then needs to produce the target under random perturbations
and cannot rely on exact values of individual pixels.
The check of the target uses the image without noise.
With `--noise_mode multigrid`, the noise is the multigrid synthesis
$\xi = \xi_1 + T_1 (\xi_2 + T_2 (\xi_3 + \dots))$
of independent Gaussian noise $\xi_l$ on all levels,
which gives blobs of all sizes instead of independent pixels.
The noise on level $l$, counted from the finest $l = 0$,
is multiplied by $q^l$ with `--noise_coarse q`.
With `--perturb σ`, the parameters themselves (the multigrid levels)
are perturbed after every step, $u_l \leftarrow u_l + \sigma_t \xi_l$,
with $\sigma_t = \sigma \, 2^{-t / H}$ halving over `--perturb_half H` steps,
or constant by default,
multiplied by $q^l$ on level $l$ with `--perturb_coarse q`.

Optional regularization of the perturbation is added to the loss:
the mean squared five-point Laplacian (`--reg_lap`),
the isotropic total variation (`--reg_tv`),
the mean squared gradient (`--reg_grad`),
and the mean squared argument of the sine $\varphi$ (`--reg_phi`).
The squared Laplacian mostly penalizes high frequencies and gives smooth images,
while the total variation gives regions of uniform color with sharp boundaries.
The squared gradient is the total variation without the square root,
which penalizes jumps quadratically and, for a wavenumber $k$, scales as $k^2$
instead of $k^4$ for the squared Laplacian.

## Results

The cases in [`cases/`](cases) use 512×512 grayscale images from gray,
`multigrid` with `cell`, `lr=0.002`, `noise=0.2`, and 1000 steps,
about 195 ms per step on one GPU.
Images at step 1000 for "A glass of wine on a table.", "A school bus on a road.",
and a longer description of a cat:

[<img src="https://pkarnakov.github.io/metamers/media/cases.png" width="800">](https://pkarnakov.github.io/metamers/media/cases.png)


| case | clean image at step 1000 | `tokens_ok` with noise |
|---|---|---|
| wine | "A glass of wine on a table.", the target, first at step 400 | 1.00, first at step 250 |
| school bus | "A distorted, grainy image of a school bus on a road." | 1.00, first at step 575 |
| cat | "A black and white photograph of a fluffy cat sitting on a windowsill, with its face mirrored across the center." | 0.97 |

The cat has the target
"A black and white photograph of a fluffy cat sitting on a windowsill,
looking straight at the camera with large round eyes, pointed ears, and long white whiskers."
Its image contains the named features, an eye, an ear, a nose, whiskers, and a window frame,
but they are scattered instead of forming one face.
For the school bus, the target is reached with noise but not without it,
where the model also describes the style of the image.

Evolution of the wine case at steps 100, 200, 300, 400, and 1000:

[<img src="https://pkarnakov.github.io/metamers/media/wine_steps.png" width="800">](https://pkarnakov.github.io/metamers/media/wine_steps.png)

| step | clean image | `tokens_ok` with noise |
|---|---|---|
| 50 | "This is a grayscale medical image, likely a CT scan or MRI, ..." | 0.56 |
| 100 | "A grainy, low-resolution image of a person's face, ..." | 0.67 |
| 200 | "A grainy, low-resolution image showing a table with several bottles and glasses." | 0.67 |
| 300 | "A grainy, low-resolution image of a person wearing a long-sleeved shirt with a wine glass on a table." | 1.00 |
| 400 | "A glass of wine on a table." | 1.00 |

A silhouette of a person appears first and is gradually replaced by the glass,
while a bottle appears at the bottom right.

### Multigrid and pixels with noise

The wine case with `multigrid` (left), `pixel` (middle),
and `pixel` with `lr=0.03` and `--reg_lap 10000` after 400 steps (right):

[<img src="https://pkarnakov.github.io/metamers/media/wine_pixel.png" width="800">](https://pkarnakov.github.io/metamers/media/wine_pixel.png)

| method | lr | regularization | steps | loss with noise | clean image |
|---|---|---|---|---|---|
| `multigrid` | 0.002 | | 1000 | 0.11 | "A glass of wine on a table." |
| `pixel` | 0.002 | | 1000 | 1.31 | "A blurry, grainy image of a person's face in the bottom right corner." |
| `pixel` | 0.03 | `--reg_lap 10000` | 400 | 0.70 | "A close-up of a wine glass with a textured, grainy background." |
| `pixel` | 0.1 | `--reg_lap 10000` | 400 | 1.21 | "A blurry, grainy image of what appears to be ancient Chinese calligraphy or seals, ..." |

With plain pixels, the image remains uniform gray with fine grain
while individual pixels reach the bounds.
Regularization alone (`--reg_lap` up to 10000 or `--reg_tv 100` with `lr=0.002`)
did not improve the loss within 225 steps, after which the runs were stopped.
With a larger learning rate and strong smoothing, some contours appear
that the model describes as a wine glass,
but the image is not recognizable.

This is a qualitative difference, which can be explained as follows.
Let $s$ be the ratio of the consistent part of the gradient in a pixel
to the part that changes with the noise.
Adam changes each parameter by about `lr` per step,
so after $t$ steps the consistent part moves a pixel by about $\text{lr} \, s \, t$
and the noise by about $\text{lr} \sqrt{t}$, like a random walk.
The consistent part dominates after $t \gtrsim 1 / s^2$ steps,
independently of the learning rate.
A component of the multigrid decomposition on a coarse level
receives the sum of the gradients over the $N$ pixels it covers,
with consistent parts adding up to $N$ and noise to $\sqrt{N}$,
so its ratio is $\sqrt{N} s$ and the number of steps is $N$ times smaller.
For the coarsest levels of a 512×512 image, $N$ is up to $10^4$–$10^5$.
Without noise, the ratio $s$ is large, and pixels converge as fast as the multigrid decomposition.

### Fourier parameterization

The wine case with `fft` after a sweep over the decay $d$ (`--fft_decay`) and the learning rate,
1000 steps each unless noted.
Images for $d = 1$ with `lr=0.001`, $d = 1.5$ with `lr=0.0001`,
$d = 1.25$ with `lr=0.0003` (the best, [`cases/wine_fft`](cases/wine_fft)),
and the same at step 2600 of a run with 5000 steps:

[<img src="https://pkarnakov.github.io/metamers/media/wine_fft.png" width="800">](https://pkarnakov.github.io/metamers/media/wine_fft.png)

| $d$ | lr | loss with noise | clean image at step 1000 |
|---|---|---|---|
| 1 | 0.0002 | 0.61 | "A vintage, grainy photograph of a wine glass with a stem, set against a textured, gray background." |
| 1 | 0.0005 | 1.18 | "A collection of six vintage-style wooden boxes, ..." |
| 1 | 0.001 | 0.40 | "A glass of wine is being poured into a glass of wine.", the target at step 975 |
| 1.25 | 0.0003 | 0.25 | "A vintage, grainy image of a wine glass on a table." |
| 1.5 | 0.00002 | 1.98 | "A blurry, monochromatic image of a face with indistinct features." |
| 1.5 | 0.0001 | 0.54 | "A grainy, monochromatic image of a wine glass and a bottle on a table." |
| 1.5 | 0.0002 | 0.25 | "A close-up of a wine glass on a table, with a dark, grainy texture." |
| 2 | 0.00001 | 1.68 | "A grainy, low-resolution black and white image of a person's face in profile, ..." |
| 2 | 0.0005 | 1.17 | "A distorted, high-contrast image of intricate, swirling patterns ..." |

With $d = 1$ and a larger learning rate,
Adam builds a grid of small saturated blobs,
which reaches the target without a recognizable image.
With $d = 2$, the images remain blurry or,
with a larger learning rate, the phase of the sine wraps around and gives stripes.
Since the gain of the lowest frequency is $512^d$,
the learning rate has to decrease with $d$.
With $d = 1.25$ or $1.5$, the images show a wine glass on a table,
smoother than with `multigrid` and on a more uniform background.

The target is reached less consistently than with `multigrid`.
In runs with 5000 steps and the same settings as [`cases/wine`](cases/wine) and [`cases/wine_fft`](cases/wine_fft),
the clean image gave exactly the target
in 49 of the first 75 checks with `multigrid` (up to step 1850, from step 650)
and in 2 of the first 105 checks with `fft` (up to step 2600),
while the loss with noise decreased in both.
With `fft`, the description adds the style,
such as "A vintage, grainy photograph of a glass of wine on a table.".
The runs differ between repetitions with the same seed
due to nondeterministic GPU kernels.

### Larger model

The wine case with Qwen3.5-2B (`--model Qwen/Qwen3.5-2B`) instead of 0.8B
and otherwise the same settings, grayscale and color.
The step takes 333 ms instead of 195 ms, and the run needs about 7.4 GB of GPU memory.
Images at step 1000 for 0.8B (left), 2B (middle), and 2B in color (right):

[<img src="https://pkarnakov.github.io/metamers/media/wine_2b.png" width="800">](https://pkarnakov.github.io/metamers/media/wine_2b.png)

| model | image | loss with noise | `tokens_ok` with noise | clean image at step 1000 |
|---|---|---|---|---|
| 0.8B | grayscale | 0.11 | 1.00, first at step 250 | "A glass of wine on a table.", the target |
| 2B | grayscale | 0.54 | 0.89, first 1.00 at step 750 | "A glass of wine sits on a textured surface, captured in a grainy, high-contrast black-and-white photograph." |
| 2B | color | 0.02 | 1.00, first at step 575 | "A colorful, pixelated image of a person wearing glasses and holding a glass of wine." |

Neither 2B run reaches the target on the clean image.
In grayscale, the 2B model draws a close-up instead of a scene:
a dark bowl of liquid fills the frame, with a bright rim and highlights on the glass,
but no stem or table.
The descriptions name a glass of dark liquid from step 300
and a wine glass from step 500,
but the model keeps describing the grain and style of the image.
In color, the model first reads "glass" as eyeglasses
("A pair of eyeglasses lies on a colorful, grainy, psychedelic background." at step 100),
later as a glass object, possibly a wine glass or goblet,
and the image remains colorful noise with a small glass and a round object.
The loss with noise is lowest for this case,
so the model is satisfied with noisy colors when the image is perturbed
but not when it is clean.

### Perturbation of the parameters

Cases with multigrid noise and perturbation of the multigrid levels
([`wine_perturb`](cases/wine_perturb), [`schoolbus_perturb`](cases/schoolbus_perturb),
[`cat_perturb`](cases/cat_perturb), [`cat_perturb_tv`](cases/cat_perturb_tv)),
512×512 grayscale, 1000 steps, the 0.8B model, and short targets.
The two cat cases use `--lr_half 200`.
Images at step 1000:

[<img src="https://pkarnakov.github.io/metamers/media/perturb.png" width="800">](https://pkarnakov.github.io/metamers/media/perturb.png)

| case | lr | noise, coarse | perturb, coarse | regularization | loss with noise | target in checks | clean image at step 1000 |
|---|---|---|---|---|---|---|---|
| wine | 0.01 | 0.3, 0.5 | 0.002, 0.5 | | 0.014 | 16 of 41, first at step 350 | "A glass of wine on a table.", the target |
| school bus | 0.005 | 0.25, 0.75 | 0.01, 0.75 | | 0.14 | 0 of 41 | "A black and white drawing of a chair on a rock." |
| cat | 0.01 | 0.4, 0.5 | 0.002, 0.5 | | 0.22 | 0 of 41 | "A black cat sitting on a wooden bench in a snowy, snowy landscape." |
| cat | 0.005 | 0.2, 0.5 | 0.002, 0.5 | `--reg_tv 0.01 --reg_lap 0.01` | 0.09 | 32 of 41, every check from step 225 | "A cat sitting on a table.", the target |

The checks are every 25 steps, including step 0.
The wine case reaches the lowest loss with noise of all wine cases
and draws a large glass with dark liquid in front of the edge of a table.
The school bus case never reaches the target:
the descriptions go through a skateboard, a lamp, a luggage rack, a wheelbarrow, and a wooden cart,
and the image shows a tall box on runners, an object with wheels but not a bus.
In the first cat case, the model draws a small black cat on a stand
in the upper left and reads the white blotches around it as snow.
The second cat case gives the target at every check from step 225 on,
with the layout fixed by then:
a table with legs and a cat with ears and a tail drawn in lines, large and in the center.
The image contains only what the target describes,
so the model has nothing else to mention.
It differs from the first cat case in the learning rate, the noise, and the regularization,
so which of them matters is still open.
This case ran after installing `flash-linear-attention`,
with 128 ms per step instead of about 194 ms for the others.

## Earlier observations

The results in this section are from 256×256 color images without noise,
with the bounds imposed by clamping unless noted otherwise.
The target is reached in tens of steps, and the images are adversarial noise.

Metamers from a gray image for "The image shows a red sports car parked on a street.":
`pixel`, `multigrid` with `cell`, and `multigrid` with `node`.

[<img src="https://pkarnakov.github.io/metamers/media/gray_car.png" width="600">](https://pkarnakov.github.io/metamers/media/gray_car.png)

Same target from a photo with perturbation bounded by `eps=0.03`:
original, `pixel`, and `multigrid` with `node`.

[<img src="https://pkarnakov.github.io/metamers/media/cats_car.png" width="600">](https://pkarnakov.github.io/metamers/media/cats_car.png)

* **No acceleration without noise.**
  All variants converge in 15–130 steps with differences comparable to the
  spread between runs, which differ due to nondeterminism on GPU.
  In ODIL, the multigrid decomposition compensates the locality of
  gradient-based optimizers on a discretization with local stencils,
  where information propagates by about one cell per iteration.
  Here, the attention layers of the model couple all image patches,
  so the gradient in each pixel already depends on the whole image.
* **Inductive bias.**
  Plain pixels give high-frequency noise,
  while the multigrid decomposition gives coherent structures on larger scales.
  For a linear model, gradient descent from zero on the components converges to
  the solution with minimal $\sum_i \lVert u_i \rVert^2$, a multilevel norm of the perturbation
  that penalizes high frequencies, as in the BPX preconditioner.
* **Bounded perturbation.**
  With `eps=0.03`, the multigrid decomposition is slower than pixels,
  since the budget per pixel is best spent on high frequencies.
* **Bounds.**
  Steps until the target from a gray image or a photo of cats,
  with `lr=0.03` halving every 5 steps:

  | run | clamping | sigmoid | sine |
  |---|---|---|---|
  | gray, cat | 25 | 65–120 | 15 |
  | cats with `eps=0.03`, car | 125 | not reached in 500 | 160 |
  | cats with `eps=0.03`, car, `pixel` | 75 | 125 | 50 |

  A sigmoid has vanishing gradients near the bounds,
  where the multigrid decomposition moves large regions at once and they get stuck.
  With clamping, pixels at the bounds receive no feedback
  while Adam keeps pushing them further.
* **Cell-based levels amplify the boundaries.**
  A unit change of a coarse component results in a perturbation
  of the image with a maximum of:

  | level (cells) | 128 | 32 | 8 | 2 |
  |---|---|---|---|---|
  | `cell`, corner | 1.56 | 2.07 | 2.20 | 2.24 |
  | `cell`, center | 0.56 | 0.56 | 0.56 | 2.24 |
  | `node`, corner or center | 1.00 | 1.00 | 1.00 | 1.00 |

  With node-based levels, each coarse value maps to a hat function with peak 1.
  With cell-based levels, interpolation to cell centers attenuates the interior,
  while the extrapolation with weights $(5, -1) / 4$ compounds over levels near the boundary.
  Since Adam changes every parameter by about `lr` per step,
  the boundary of the image receives about four times larger updates than the interior,
  which shows as bands near the edges of the images.
  This may also explain the slower convergence of cell-based discretizations
  observed for the Poisson equation in mODIL.
* **Learning rate.**
  Each level of the multigrid decomposition changes by about `lr` per step,
  so a larger learning rate gives more vivid images that saturate at the bounds.
  Saturation is decided in the first steps,
  and a large initial learning rate that decays fast (`--lr_half`, `--lr_min`)
  keeps vivid colors with fewer saturated pixels.
  From a gray image for "This is a picture of a cat." with `node`:
  constant `lr=0.003`, constant `lr=0.03`, and `lr=0.03` halving every 5 steps.

  [<img src="https://pkarnakov.github.io/metamers/media/gray_lr.png" width="600">](https://pkarnakov.github.io/metamers/media/gray_lr.png)

  From a photo of cats for "The image shows a red sports car parked on a street."
  with `eps=0.2`: original, constant `lr=0.003`, constant `lr=0.03`,
  and `lr=0.03` halving every 5 steps.

  [<img src="https://pkarnakov.github.io/metamers/media/cats_eps02_lr.png" width="600">](https://pkarnakov.github.io/metamers/media/cats_eps02_lr.png)
* **Regularization without noise.**
  From a gray image after 200 steps, `--reg_lap 1000` gives smooth blobs,
  and `--reg_tv 300` gives large regions of uniform color with sharp boundaries,
  both still producing the target, while `--reg_tv 1000` does not.
