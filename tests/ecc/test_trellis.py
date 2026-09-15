import torch

from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import (
    codebook_tensor,
    codeword_log_likelihoods,
    decode_single,
    grouped_nll_loss,
    nll_loss,
)
from neo_toolkit.ecc.trellis import (
    build_trellis,
    trellis_decode_bits,
    trellis_grouped_nll_loss,
    trellis_log_z,
    trellis_nll_loss,
)

CODEBOOK = codebook_tensor(NEO_20_10)
TRELLIS = build_trellis(NEO_20_10)


def _random_logits(*shape: int, seed: int = 0, scale: float = 2.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, NEO_20_10.n, generator=g) * scale


def test_active_set_matches_known_worst_case():
    # Documents the finding the README explains: this code's H is dense, so
    # the natural-order trellis reaches the full 2**10 state space at the
    # midpoint -- no compression versus brute force. If this ever changes
    # (e.g. the code is swapped out), the width claim in the module
    # docstrings should be revisited too.
    widths = [1] + [s.n_out for s in TRELLIS.steps]
    assert max(widths) == 1024
    assert widths[10] == 1024  # midpoint boundary


def test_trellis_log_z_matches_brute_force_no_channel():
    logits = _random_logits(6, seed=1)

    trellis_z = trellis_log_z(logits, TRELLIS)
    brute_z = codeword_log_likelihoods(logits, CODEBOOK).logsumexp(dim=-1)

    assert torch.allclose(trellis_z, brute_z, atol=1e-4)


def test_trellis_log_z_matches_brute_force_with_channel():
    channel = ZChannel(p=0.2)
    logits = _random_logits(6, seed=2)

    trellis_z = trellis_log_z(logits, TRELLIS, channel=channel)
    brute_z = codeword_log_likelihoods(logits, CODEBOOK, channel=channel).logsumexp(dim=-1)

    assert torch.allclose(trellis_z, brute_z, atol=1e-4)


def test_trellis_log_z_matches_brute_force_grouped_shape():
    logits = _random_logits(3, 5, seed=3)  # (batch, J, n)

    trellis_z = trellis_log_z(logits, TRELLIS)
    brute_z = codeword_log_likelihoods(logits, CODEBOOK).logsumexp(dim=-1)

    assert trellis_z.shape == (3, 5)
    assert torch.allclose(trellis_z, brute_z, atol=1e-4)


def test_trellis_decode_matches_brute_force_map_decode():
    logits = _random_logits(20, seed=4)

    trellis_bits = trellis_decode_bits(logits, TRELLIS)
    brute_indices = decode_single(logits, CODEBOOK)
    brute_bits = CODEBOOK[brute_indices]

    assert torch.equal(trellis_bits, brute_bits)


def test_trellis_decode_matches_brute_force_map_decode_with_channel():
    channel = ZChannel(p=0.2)
    logits = _random_logits(20, seed=5)

    trellis_bits = trellis_decode_bits(logits, TRELLIS, channel=channel)
    brute_indices = decode_single(logits, CODEBOOK, channel=channel)
    brute_bits = CODEBOOK[brute_indices]

    assert torch.equal(trellis_bits, brute_bits)


def test_trellis_decode_recovers_confident_codeword():
    target = 682
    bits = CODEBOOK[target]
    logits = (2.0 * bits - 1.0) * 8.0

    decoded = trellis_decode_bits(logits, TRELLIS)

    assert torch.equal(decoded, bits)


def test_trellis_nll_loss_matches_brute_force():
    logits = _random_logits(10, seed=7)
    targets = torch.randint(0, 1024, (10,), generator=torch.Generator().manual_seed(8))

    trellis_loss = trellis_nll_loss(logits, targets, CODEBOOK, TRELLIS)
    brute_loss = nll_loss(logits, targets, CODEBOOK)

    assert torch.allclose(trellis_loss, brute_loss, atol=1e-4)


def test_trellis_grouped_nll_loss_matches_brute_force():
    logits = _random_logits(6, 5, seed=9)  # (batch, J, n)
    targets = torch.randint(0, 1024, (6,), generator=torch.Generator().manual_seed(10))
    mask = (torch.rand(6, 5, generator=torch.Generator().manual_seed(11)) > 0.3).float()
    mask[:, 0] = 1.0  # every group keeps at least one real observation

    trellis_loss = trellis_grouped_nll_loss(logits, targets, CODEBOOK, mask, TRELLIS)
    brute_loss = grouped_nll_loss(logits, targets, CODEBOOK, mask)

    assert torch.allclose(trellis_loss, brute_loss, atol=1e-4)


def test_trellis_grouped_nll_loss_matches_brute_force_with_channel():
    channel = ZChannel(p=0.2)
    logits = _random_logits(6, 5, seed=12)
    targets = torch.randint(0, 1024, (6,), generator=torch.Generator().manual_seed(13))
    mask = torch.ones(6, 5)

    trellis_loss = trellis_grouped_nll_loss(logits, targets, CODEBOOK, mask, TRELLIS, channel=channel)
    brute_loss = grouped_nll_loss(logits, targets, CODEBOOK, mask, channel=channel)

    assert torch.allclose(trellis_loss, brute_loss, atol=1e-4)


def test_trellis_log_z_is_differentiable():
    logits = _random_logits(4, seed=6).requires_grad_(True)

    trellis_log_z(logits, TRELLIS).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
