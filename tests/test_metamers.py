"""
Tests of the parts of metamers.py that do not need the model weights.
The processor of the model (a few MB, without the weights)
is downloaded from the Hub unless cached.
"""

import functools

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

import metamers
from metamers import Loc

MODEL = "Qwen/Qwen3.5-0.8B"


@functools.cache
def processor():
    return AutoProcessor.from_pretrained(MODEL)


@pytest.mark.parametrize("n_out, n_in", [(256, 64), (256, 96), (64, 256), (100, 100)])
def test_bicubic_matches_pil(n_out, n_in):
    a = np.random.default_rng(0).random((n_in, n_in), dtype=np.float32)
    ref = np.asarray(Image.fromarray(a, mode="F").resize((n_out, n_out), Image.Resampling.BICUBIC))
    m = metamers.pil_bicubic_matrix(n_out, n_in)
    res = (m @ torch.from_numpy(a) @ m.T).numpy()
    np.testing.assert_allclose(res, ref, atol=1e-5)


@pytest.mark.parametrize("size", [64, 256, 320])
@pytest.mark.parametrize("grayscale", [False, True])
def test_patchify_matches_processor(size, grayscale):
    c = 1 if grayscale else 3
    # Quantized to 8 bits as the image passed to the processor.
    x = (torch.rand(c, size, size, generator=torch.Generator().manual_seed(0)) * 255).round() / 255
    ref = processor().image_processor(images=[metamers.to_pil(x)], return_tensors="pt")
    pv, grid = metamers.Patchify(processor().image_processor)(x)
    assert torch.equal(grid, ref["image_grid_thw"])
    # PIL also rounds to 8 bits after each pass of the resize.
    tol = 2e-2 if size < 256 else 1e-5
    torch.testing.assert_close(pv, ref["pixel_values"], atol=tol, rtol=0)


def linear_field(c, h, w, loc, scale=1.0):
    """Linear function of the coordinates of cell centers or nodes, multiplied by scale."""
    offset = 0.5 if loc == Loc.CELL else 0
    y = (torch.arange(h, dtype=torch.float64)[:, None] + offset) * scale
    x = (torch.arange(w, dtype=torch.float64)[None, :] + offset) * scale
    return torch.stack([1 + k + 2 * x - 3 * y for k in range(c)])


@pytest.mark.parametrize("loc", list(Loc))
def test_interp_to_finer_exact_for_linear(loc):
    h, w = 8, 5
    fine = metamers.interp_to_finer(linear_field(2, h, w, loc), loc)
    if loc == Loc.CELL:
        assert fine.shape == (2, 2 * h, 2 * w)
    else:
        assert fine.shape == (2, 2 * (h - 1) + 1, 2 * (w - 1) + 1)
    # The finer grid has half the spacing.
    torch.testing.assert_close(fine, linear_field(2, fine.shape[1], fine.shape[2], loc, scale=0.5))


@pytest.mark.parametrize("loc", list(Loc))
def test_multigrid_levels(loc):
    mg = metamers.Multigrid((1, 32, 32), 4, loc, "cpu")
    extra = 1 if loc == Loc.NODE else 0
    assert [tuple(t.shape) for t in mg.terms] == [(1, n + extra, n + extra) for n in (32, 16, 8, 4)]
    with torch.no_grad():
        mg.terms[-1].fill_(1)
    # A constant on the coarsest level gives the same constant on the image.
    torch.testing.assert_close(mg(), torch.ones(1, 32, 32))


def test_fourier_decay_zero_is_pixels():
    u = torch.rand(1, 16, 16, generator=torch.Generator().manual_seed(0))
    f = metamers.Fourier(u.shape, 0, "cpu")
    with torch.no_grad():
        f.terms[0].copy_(torch.view_as_real(torch.fft.rfft2(u, norm="ortho")))
    torch.testing.assert_close(f(), u)


def test_regularizers_vanish_on_constant():
    u = torch.full((1, 16, 16), 0.3)
    assert metamers.laplacian(u).abs().max() == 0
    assert metamers.squared_gradient(u) == 0
    assert metamers.total_variation(u) < 1e-5


def test_laplacian_vanishes_on_linear():
    u = linear_field(1, 16, 16, Loc.NODE)
    torch.testing.assert_close(metamers.laplacian(u), torch.zeros(1, 14, 14, dtype=torch.float64))
