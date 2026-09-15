import torch

from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import (
    codebook_tensor,
    codeword_log_likelihoods,
    decode_grouped,
    decode_single,
    grouped_nll_loss,
    nll_loss,
)

CODEBOOK = codebook_tensor(NEO_20_10)


def _confident_logits(codeword_index: int, *, magnitude: float = 8.0) -> torch.Tensor:
    bits = CODEBOOK[codeword_index]
    return (2.0 * bits - 1.0) * magnitude


def test_codeword_log_likelihoods_shape_batched():
    logits = torch.zeros(5, NEO_20_10.n)

    scores = codeword_log_likelihoods(logits, CODEBOOK)

    assert scores.shape == (5, 1024)


def test_codeword_log_likelihoods_shape_grouped():
    logits = torch.zeros(3, 7, NEO_20_10.n)  # (batch, J, n)

    scores = codeword_log_likelihoods(logits, CODEBOOK)

    assert scores.shape == (3, 7, 1024)


def test_decode_single_recovers_codeword_from_confident_logits():
    target = 682
    logits = _confident_logits(target).unsqueeze(0)

    decoded = decode_single(logits, CODEBOOK)

    assert decoded.item() == target


def test_decode_grouped_recovers_codeword_from_noisy_repeats():
    target = 99
    clean_bits = CODEBOOK[target]
    rng = torch.Generator().manual_seed(0)
    # 8 repeats, each with independent random noise on top of a confident signal
    noise = torch.randn(1, 8, NEO_20_10.n, generator=rng) * 0.5
    logits = ((2.0 * clean_bits - 1.0) * 3.0).unsqueeze(0).unsqueeze(0) + noise
    mask = torch.ones(1, 8)

    decoded = decode_grouped(logits, CODEBOOK, mask)

    assert decoded.item() == target


def test_grouped_decode_ignores_masked_out_observations():
    target = 3
    clean_bits = CODEBOOK[target]
    confident = ((2.0 * clean_bits - 1.0) * 8.0).unsqueeze(0).unsqueeze(0)
    garbage = torch.zeros(1, 1, NEO_20_10.n)  # would be totally uninformative if counted
    logits = torch.cat([confident, garbage], dim=1)
    mask = torch.tensor([[1.0, 0.0]])

    decoded = decode_grouped(logits, CODEBOOK, mask)

    assert decoded.item() == target


def test_nll_loss_gradient_descent_converges_to_target():
    target_index = 42
    target = torch.tensor([target_index])
    logits = torch.zeros(1, NEO_20_10.n, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)

    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        loss = nll_loss(logits, target, CODEBOOK)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-3
    with torch.no_grad():
        assert decode_single(logits, CODEBOOK).item() == target_index


def test_grouped_nll_loss_has_nonzero_gradient():
    target = torch.tensor([0])
    logits = torch.randn(1, 4, NEO_20_10.n, requires_grad=True)
    mask = torch.ones(1, 4)

    loss = grouped_nll_loss(logits, target, CODEBOOK, mask)
    loss.backward()

    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_channel_aware_is_less_punishing_of_flippable_bits_than_direct_bits():
    # Under a Z-channel, an observed 0 is ambiguous (could be a corrupted 1),
    # so a channel-aware decoder should score a codeword with a 1 there less
    # harshly than a direct-bits decoder would.
    codeword_with_one_at_0 = next(i for i in range(1024) if CODEBOOK[i, 0] == 1)
    logits = torch.full((1, NEO_20_10.n), -8.0)  # detector confidently says "all zeros"

    direct_scores = codeword_log_likelihoods(logits, CODEBOOK)
    channel_scores = codeword_log_likelihoods(logits, CODEBOOK, channel=ZChannel(p=0.3))

    idx = codeword_with_one_at_0
    all_zero_idx = 0
    direct_gap = direct_scores[0, all_zero_idx] - direct_scores[0, idx]
    channel_gap = channel_scores[0, all_zero_idx] - channel_scores[0, idx]
    assert channel_gap < direct_gap
