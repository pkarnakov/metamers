#!/usr/bin/env python3
"""
Image metamers for a vision-language model (Qwen3.5).

Finds an image for which the model answers a prompt with a given target
string. The image is optimized with Adam, using gradients of the
teacher-forced cross-entropy of the target tokens, either in plain pixels,
through a multigrid decomposition (sum of components on coarser grids),
or through the Fourier spectrum scaled by a power of the inverse frequency.
"""

import argparse
import functools
import json
import os
import time
from enum import StrEnum

# Cached models only, no requests to the Hub; HF_HUB_OFFLINE=0 to download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Compiled kernels are kept across reboots, unlike the default in /tmp.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import logging

# Warnings about reference implementations of linear attention, which are
# as fast as flash-linear-attention for sequences of this length.
logging.get_logger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)
# Warnings from torch.compile: no autotuning of matrix products on GPUs with few SMs,
# and graph breaks at Tensor.item() in the vision model, which reads the grid size.
logging.get_logger("torch._inductor.utils").setLevel(logging.ERROR)
logging.get_logger("torch._dynamo.variables.tensor").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TF32 for float32 matrix products, e.g. in linear attention, since the model is in bfloat16.
torch.set_float32_matmul_precision("high")


class Method(StrEnum):
    PIXEL = "pixel"
    MULTIGRID = "multigrid"
    FFT = "fft"


class Loc(StrEnum):
    CELL = "cell"
    NODE = "node"


class Noise(StrEnum):
    PIXEL = "pixel"
    MULTIGRID = "multigrid"


# Learning rate and steps over which it halves, zero for constant.
# Multigrid starts large for vivid large-scale structures and decays fast,
# since saturation of pixels is decided in the first steps.
# With fft, the steps of Adam on all coefficients add up in pixels,
# so the first step at 256x256 changes pixels as much as multigrid with a 10 times larger rate.
DEFAULT_LR = {Method.PIXEL: 0.01, Method.MULTIGRID: 0.03, Method.FFT: 0.003}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="Hugging Face model name")
    parser.add_argument("--target", default="This is a picture of a cat.", help="Desired model output")
    parser.add_argument("--prompt", default="Describe this image in one sentence.", help="Text prompt after the image")
    parser.add_argument(
        "--size", type=int, default=256, help="Image size, multiple of 32, upscaled by the processor to at least 256"
    )
    parser.add_argument("--init", default="gray", help="'gray', 'noise', or path to an image")
    parser.add_argument("--grayscale", action="store_true", help="Optimize a grayscale image")
    parser.add_argument("--eps", type=float, default=None, help="Optional L-inf bound around the initial image")
    parser.add_argument("--steps", type=int, default=500, help="Maximum number of optimization steps")
    parser.add_argument(
        "--method",
        type=Method,
        choices=list(Method),
        default=Method.MULTIGRID,
        help="Image parameterization: plain pixels, multigrid decomposition, or Fourier spectrum",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate, None for " + ", ".join(f"{lr} with {m}" for m, lr in DEFAULT_LR.items()),
    )
    parser.add_argument(
        "--lr_half",
        type=float,
        default=0,
        help="Steps over which the learning rate halves, zero for constant",
    )
    parser.add_argument(
        "--reg_lap", type=float, default=0, help="Weight of the mean squared Laplacian of the perturbation"
    )
    parser.add_argument(
        "--reg_tv", type=float, default=0, help="Weight of the isotropic total variation of the perturbation"
    )
    parser.add_argument(
        "--reg_grad", type=float, default=0, help="Weight of the mean squared gradient of the perturbation"
    )
    parser.add_argument(
        "--noise", type=float, default=0, help="Standard deviation of Gaussian noise added to the image in the loss"
    )
    parser.add_argument(
        "--noise_half",
        type=float,
        default=0,
        help="Steps over which the magnitude of --noise halves, zero for constant",
    )
    parser.add_argument(
        "--noise_coarse",
        type=float,
        default=1,
        help="Factor of the noise on each coarser level with --noise_mode multigrid",
    )
    parser.add_argument(
        "--noise_mode",
        type=Noise,
        choices=list(Noise),
        default=Noise.PIXEL,
        help="Noise independent in pixels, or the multigrid synthesis of independent noise on all levels,"
        " with equal amplitude per level and the same --mg_loc, independently of --method",
    )
    parser.add_argument(
        "--reg_phi", type=float, default=0, help="Weight of the mean squared phase, the argument of the sine"
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0,
        help="Standard deviation of Gaussian noise added to the optimized parameters (multigrid levels) after every step",
    )
    parser.add_argument(
        "--perturb_half",
        type=float,
        default=0,
        help="Steps over which the magnitude of --perturb halves, zero for constant",
    )
    parser.add_argument(
        "--perturb_coarse",
        type=float,
        default=1,
        help="Factor of --perturb on each coarser multigrid level",
    )
    parser.add_argument("--lr_min", type=float, default=0.001, help="Lower bound of the decaying learning rate")
    parser.add_argument(
        "--mg_loc",
        type=Loc,
        choices=list(Loc),
        default=Loc.CELL,
        help="Multigrid values in cells (pixels) or nodes (pixel corners plus one extra row and column, cropped)",
    )
    parser.add_argument("--mg_nlvl", type=int, default=None, help="Number of multigrid levels, None for maximum")
    parser.add_argument(
        "--fft_decay",
        type=float,
        default=1,
        help="Spectrum with fft scaled by the inverse frequency to this power, zero for white",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model in the loss, a few minutes for new shapes, e.g. target length, cached afterwards",
    )
    parser.add_argument(
        "--stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop once the target is reached instead of running all steps",
    )
    parser.add_argument("--check_every", type=int, default=25, help="Steps between generation checks")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum length of generated output, None for the target length plus 16 but at least 48",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", default="out", help="Output directory")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR[args.method]
    return args


def load_init(init, size, grayscale=False):
    """
    Returns initial image as tensor of shape (C, size, size) in [0, 1],
    with C=1 if grayscale, otherwise C=3.
    """
    c = 1 if grayscale else 3
    if init == "gray":
        return torch.full((c, size, size), 0.5)
    if init == "noise":
        return torch.rand(c, size, size)
    img = Image.open(init).convert("L" if grayscale else "RGB").resize((size, size), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(img).copy()).view(size, size, c).permute(2, 0, 1).float() / 255


def to_pil(x):
    """Quantizes image tensor (C, H, W) in [0, 1] with C=1 or 3 to 8-bit RGB PIL image."""
    a = (x.detach().expand(3, -1, -1).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


@functools.cache
def pil_bicubic_matrix(n_out, n_in):
    """
    Matrix of resampling from n_in to n_out values
    as PIL.Image.resize() with BICUBIC, including antialiasing on downscaling.
    """
    scale = n_in / n_out
    fscale = max(scale, 1)
    support = 2 * fscale

    def kernel(x, a=-0.5):
        x = abs(x)
        if x < 1:
            return ((a + 2) * x - (a + 3)) * x * x + 1
        if x < 2:
            return (((x - 5) * x + 8) * x - 4) * a
        return 0

    res = torch.zeros(n_out, n_in)
    for i in range(n_out):
        center = (i + 0.5) * scale
        jmin, jmax = max(int(center - support + 0.5), 0), min(int(center + support + 0.5), n_in)
        w = torch.tensor([kernel((j - center + 0.5) / fscale) for j in range(jmin, jmax)])
        res[i, jmin:jmax] = w / w.sum()
    return res


class Patchify:
    """Differentiable version of Qwen2VLImageProcessor: resize, normalize, and patchify."""

    def __init__(self, image_processor):
        ip = image_processor
        self.patch = ip.patch_size
        self.merge = ip.merge_size
        self.temporal = ip.temporal_patch_size
        self.mean = torch.tensor(ip.image_mean).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std).view(3, 1, 1)
        self.min_pixels = ip.size["shortest_edge"]
        self.max_pixels = ip.size["longest_edge"]

    def __call__(self, x):
        """
        x: Tensor, shape (C, H, W) with C=1 or 3, values in [0, 1]
        Returns: pixel_values (seq_len, patch_dim), image_grid_thw (1, 3)
        """
        x = x.expand(3, -1, -1)  # grayscale to RGB
        p, m, t = self.patch, self.merge, self.temporal
        # Resize to a multiple of p * m within the pixel limits, e.g. at least 256x256.
        h, w = smart_resize(x.shape[1], x.shape[2], p * m, self.min_pixels, self.max_pixels)
        if (h, w) != x.shape[1:]:
            mh = pil_bicubic_matrix(h, x.shape[1]).to(x)
            mw = pil_bicubic_matrix(w, x.shape[2]).to(x)
            # Horizontal then vertical, clipped after each as in PIL.
            x = (mh @ (x @ mw.T).clamp(0, 1)).clamp(0, 1)
        c = x.shape[0]
        gh, gw = h // p, w // p
        x = (x - self.mean.to(x)) / self.std.to(x)
        x = x.reshape(c, gh // m, m, p, gw // m, m, p)
        x = x.permute(1, 4, 2, 5, 0, 3, 6)  # [gh/m, gw/m, m, m, c, p, p]
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, t, -1, -1)
        pixel_values = x.reshape(gh * gw, c * t * p * p)
        grid = torch.tensor([[1, gh, gw]], device=x.device)
        return pixel_values, grid


def laplacian(u):
    """Five-point Laplacian in pixel units, interior pixels only. u: (C, H, W)"""
    return u[:, :-2, 1:-1] + u[:, 2:, 1:-1] + u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 4 * u[:, 1:-1, 1:-1]


def total_variation(u, eps=1e-6):
    """Isotropic total variation per pixel, forward differences. u: (C, H, W)"""
    ux = u[:, :-1, 1:] - u[:, :-1, :-1]
    uy = u[:, 1:, :-1] - u[:, :-1, :-1]
    return (ux.square() + uy.square() + eps**2).sqrt().mean()


def squared_gradient(u):
    """Mean squared gradient per pixel, forward differences as in total_variation(). u: (C, H, W)"""
    ux = u[:, :-1, 1:] - u[:, :-1, :-1]
    uy = u[:, 1:, :-1] - u[:, :-1, :-1]
    return (ux.square() + uy.square()).mean()


def interp_to_finer(u, loc):
    """
    Interpolates a field to a grid refined twice in each direction,
    as `odil.core.interp_to_finer()`.

    u: Tensor, shape (C, H, W)
    loc: Loc
        CELL: values in cell centers, linear extrapolation at the boundaries,
              new size 2 * H;
        NODE: values in nodes, new size 2 * (H - 1) + 1.
    """
    for dim in (1, 2):
        u = u.movedim(dim, 0)
        if loc == Loc.CELL:
            upad = torch.cat([2 * u[:1] - u[1:2], u, 2 * u[-1:] - u[-2:-1]])
            left, mid, right = upad[:-2], upad[1:-1], upad[2:]
            fine = torch.stack([0.25 * left + 0.75 * mid, 0.75 * mid + 0.25 * right], dim=1)
            u = fine.reshape((-1,) + fine.shape[2:])
        else:
            fine = torch.stack([u[:-1], 0.5 * (u[:-1] + u[1:])], dim=1)
            u = torch.cat([fine.reshape((-1,) + fine.shape[2:]), u[-1:]])
        u = u.movedim(0, dim)
    return u


class Multigrid:
    """
    Multigrid decomposition: a field on the fine grid as the sum of components
    on grids coarsened twice per level, u = u0 + P(u1 + P(u2 + ...)).
    With values in nodes, the fine grid has one extra row and column
    that are cropped from the result.
    """

    def __init__(self, shape, nlvl, loc, device, requires_grad=True):
        c, h, w = shape
        self.shape, self.loc = shape, loc
        extra = 1 if loc == Loc.NODE else 0
        self.terms = [
            torch.zeros(c, (h >> l) + extra, (w >> l) + extra, device=device, requires_grad=requires_grad)
            for l in range(nlvl)
        ]

    def __call__(self):
        res = self.terms[-1]
        for term in reversed(self.terms[:-1]):
            res = term + interp_to_finer(res, self.loc)
        _, h, w = self.shape
        return res[:, :h, :w]


class Fourier:
    """
    Field as the inverse real FFT of a spectrum scaled by 1 / f^decay,
    with the frequency f in cycles per pixel bounded below by the lowest
    nonzero frequency. The transform is orthonormal, so decay 0 is equivalent
    to pixels, and with decay 1 the zero frequency of a square changes all
    pixels as the coarsest multigrid component.
    """

    def __init__(self, shape, decay, device):
        c, h, w = shape
        self.shape = shape
        fy = torch.fft.fftfreq(h, device=device)[:, None]
        fx = torch.fft.rfftfreq(w, device=device)[None, :]
        freq = (fy.square() + fx.square()).sqrt().clamp(min=1 / max(h, w))
        self.scale = freq ** (-decay)
        # Real and imaginary parts.
        self.terms = [torch.zeros(c, h, w // 2 + 1, 2, device=device, requires_grad=True)]

    def __call__(self):
        _, h, w = self.shape
        spectrum = torch.view_as_complex(self.terms[0]) * self.scale
        return torch.fft.irfft2(spectrum, s=(h, w), norm="ortho")


def build_messages(prompt):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


@torch.no_grad()
def generate(model, processor, image, prompt, max_new_tokens):
    """Runs the standard (non-differentiable) pipeline on a PIL image."""
    text = processor.apply_chat_template(
        build_messages(prompt), add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    batch = processor(text=[text], images=[image], return_tensors="pt").to(device)
    # The model config only has <|endoftext|> as EOS, while chat turns end with <|im_end|>.
    eos = processor.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos)
    return processor.decode(out[0, batch["input_ids"].shape[1] :], skip_special_tokens=True)


def config_yaml(args):
    """Arguments as YAML, one line per argument."""
    config = {k: str(v) if isinstance(v, StrEnum) else v for k, v in vars(args).items()}
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=float("inf"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    assert args.size % 32 == 0, "size must be a multiple of patch_size * merge_size = 32"

    processor = AutoProcessor.from_pretrained(args.model)
    if args.max_new_tokens is None:
        args.max_new_tokens = max(48, len(processor.tokenizer(args.target).input_ids) + 16)
    config = config_yaml(args)
    print(config, end="", flush=True)
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        f.write(config)

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval().requires_grad_(False)
    patchify = Patchify(processor.image_processor)

    x0 = load_init(args.init, args.size, args.grayscale).to(device)
    print(f"initial output: {generate(model, processor, to_pil(x0), args.prompt, args.max_new_tokens)!r}")

    # Prompt tokens followed by target tokens and end-of-turn.
    text = processor.apply_chat_template(
        build_messages(args.prompt), add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    batch = processor(text=[text], images=[to_pil(x0)], return_tensors="pt").to(device)
    target_ids = processor.tokenizer(args.target + "<|im_end|>", return_tensors="pt").input_ids.to(device)
    input_ids = torch.cat([batch["input_ids"], target_ids], dim=1)
    n_target = target_ids.shape[1]
    extra = {}
    if "mm_token_type_ids" in batch:
        extra["mm_token_type_ids"] = F.pad(batch["mm_token_type_ids"], (0, n_target))

    # Check that differentiable preprocessing matches the processor
    # up to rounding to 8 bits, which PIL also does after each pass of the resize.
    pv, grid = patchify(x0)
    err = (pv - batch["pixel_values"]).abs().max().item()
    assert torch.equal(grid.cpu(), batch["image_grid_thw"].cpu()) and err < 2e-2, f"patchify mismatch: {err}"

    def forward(pv, grid):
        # Logits only at the last n_target + 1 positions, where position i predicts token i + 1.
        return model(
            input_ids=input_ids, pixel_values=pv, image_grid_thw=grid, logits_to_keep=n_target + 1, **extra
        ).logits

    if args.compile:
        forward = torch.compile(forward)

    def loss_fn(x):
        logits = forward(*patchify(x))
        logits = logits[0, :n_target].float()
        tokens_ok = (logits.argmax(-1) == target_ids[0]).float().mean().item()
        return F.cross_entropy(logits, target_ids[0]), tokens_ok

    # Image as a sine of the initial phase plus a perturbation,
    # which keeps the image within the bounds. One level means plain pixels.
    # Levels while the size halves exactly, down to 2 cells for a power of two.
    nlvl_max = (args.size & -args.size).bit_length() - 1
    if args.method == Method.FFT:
        delta = Fourier(x0.shape, args.fft_decay, device)
    else:
        nlvl = min(args.mg_nlvl or nlvl_max, nlvl_max) if args.method == Method.MULTIGRID else 1
        delta = Multigrid(x0.shape, nlvl, args.mg_loc, device)
        print(f"multigrid levels: {[tuple(t.shape[1:]) for t in delta.terms]}")
    optimizer = torch.optim.Adam(delta.terms, lr=args.lr)
    half = args.lr_half or float("inf")
    floor = min(1, args.lr_min / args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.5 ** (step / half), floor))
    lo, hi = torch.zeros_like(x0), torch.ones_like(x0)
    if args.eps is not None:
        lo, hi = (x0 - args.eps).clamp(0, 1), (x0 + args.eps).clamp(0, 1)
    center, radius = (hi + lo) / 2, (hi - lo) / 2
    # Initial phase, kept away from the bounds for nonzero gradients.
    z0 = torch.asin(((x0 - center) / radius).clamp(-1 + 1e-3, 1 - 1e-3))

    def phase():
        # Scaled such that the image changes as the perturbation around the midpoint.
        return z0 + delta() / radius

    def image(phi):
        return center + radius * torch.sin(phi)

    if args.noise_mode == Noise.MULTIGRID:
        noise_mg = Multigrid(x0.shape, nlvl_max, args.mg_loc, device, requires_grad=False)

        def noise_like(x):
            for level, term in enumerate(noise_mg.terms):
                term.normal_().mul_(args.noise_coarse**level)
            return noise_mg()
    else:
        noise_like = torch.randn_like

    log = open(os.path.join(args.out, "train.log"), "w")
    # Records for the viewer: start with the id of the run, one per check, done at the end.
    jsonl = open(os.path.join(args.out, "train.jsonl"), "w")

    def record(**kwargs):
        jsonl.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
        jsonl.flush()

    t0 = time.time()
    record(event="start", run=f"{t0:.3f}")
    # Time of optimization steps since the last check, without the checks.
    step_time, step_count = 0.0, 0
    reached = False
    for step in range(args.steps + 1):
        t_step = time.time()
        optimizer.zero_grad()
        phi = phase()
        x = image(phi)
        # Fresh noise every step, for robustness to small changes of the image.
        noise = args.noise * 0.5 ** (step / args.noise_half) if args.noise_half else args.noise
        loss, tokens_ok = loss_fn(x + noise * noise_like(x))
        reg = (
            args.reg_lap * laplacian(x - x0).square().mean()
            + args.reg_tv * total_variation(x - x0)
            + args.reg_grad * squared_gradient(x - x0)
            + args.reg_phi * phi.square().mean()
        )
        (loss + reg).backward()
        optimizer.step()
        scheduler.step()
        if args.perturb:
            sigma = args.perturb * 0.5 ** (step / args.perturb_half) if args.perturb_half else args.perturb
            with torch.no_grad():
                for level, term in enumerate(delta.terms):
                    term.add_(sigma * args.perturb_coarse**level * torch.randn_like(term))
        with torch.no_grad():
            x = image(phase())
        step_time += time.time() - t_step
        step_count += 1

        if step % args.check_every == 0 or step == args.steps:
            img = to_pil(x)
            img.save(os.path.join(args.out, f"metamer_{step:05d}.png"))
            output = generate(model, processor, img, args.prompt, args.max_new_tokens)
            linf = (x - x0).abs().max().item()
            msg = (
                f"step={step:5d} loss={loss.item():.4f} reg={reg.item():.4f} tokens_ok={tokens_ok:.2f}"
                f" linf={linf:.3f} time={time.time() - t0:.0f}s ms/step={step_time / step_count * 1000:.0f}"
                f" output={output!r}"
            )
            record(
                step=step,
                loss=loss.item(),
                reg=reg.item(),
                tokens_ok=tokens_ok,
                linf=linf,
                time=time.time() - t0,
                ms_step=step_time / step_count * 1000,
                output=output,
                image=f"metamer_{step:05d}.png",
            )
            step_time, step_count = 0.0, 0
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            if output == args.target:
                if not reached:
                    print("target reached (8-bit image, standard pipeline)")
                    reached = True
                # Latest image that produces the target.
                img.save(os.path.join(args.out, "metamer.png"))
                if args.stop:
                    break
    record(event="done")


if __name__ == "__main__":
    main()
