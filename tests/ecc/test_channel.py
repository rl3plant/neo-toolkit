import numpy as np
import torch

from neo_toolkit.ecc.channel import BinarySymmetricChannel, ZChannel


def test_zchannel_never_flips_zero_to_one():
    rng = np.random.default_rng(0)
    bits = np.zeros(10_000, dtype=np.uint8)

    corrupted = ZChannel(p=0.9).corrupt(bits, rng)

    assert np.all(corrupted == 0)


def test_zchannel_drops_ones_at_roughly_rate_p():
    rng = np.random.default_rng(0)
    bits = np.ones(10_000, dtype=np.uint8)

    corrupted = ZChannel(p=0.3).corrupt(bits, rng)

    drop_rate = 1.0 - corrupted.mean()
    assert abs(drop_rate - 0.3) < 0.02


def test_bsc_flips_at_roughly_rate_p_in_both_directions():
    rng = np.random.default_rng(0)
    bits = (np.arange(10_000) % 2).astype(np.uint8)

    corrupted = BinarySymmetricChannel(p=0.2).corrupt(bits, rng)

    flip_rate = (corrupted != bits).mean()
    assert abs(flip_rate - 0.2) < 0.02


def test_zchannel_zero_noise_likelihood_is_identity():
    q1 = torch.tensor([0.1, 0.5, 0.9])

    psi1, psi0 = ZChannel(p=0.0).likelihood_transform(q1)

    assert torch.allclose(psi1, q1)
    assert torch.allclose(psi0, 1.0 - q1)


def test_bsc_full_noise_likelihood_is_uninformative():
    q1 = torch.tensor([0.1, 0.5, 0.9])

    psi1, psi0 = BinarySymmetricChannel(p=0.5).likelihood_transform(q1)

    assert torch.allclose(psi1, torch.full_like(q1, 0.5))
    assert torch.allclose(psi0, torch.full_like(q1, 0.5))
