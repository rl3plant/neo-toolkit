import torch

from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import codebook_tensor
from neo_toolkit.ecc.metrics import compute_metrics, oracle_metrics

CODEBOOK = codebook_tensor(NEO_20_10)


def _confident_logits_for(index: int, *, j: int, magnitude: float = 8.0) -> torch.Tensor:
    bits = CODEBOOK[index]
    single = (2.0 * bits - 1.0) * magnitude
    return single.unsqueeze(0).repeat(j, 1)


def test_compute_metrics_zero_error_for_perfectly_confident_correct_logits():
    target = 17
    logits = _confident_logits_for(target, j=5).unsqueeze(0)
    clean_bits = CODEBOOK[target].unsqueeze(0)
    target_indices = torch.tensor([target])
    group_mask = torch.ones(1, 5)

    metrics = compute_metrics(logits, clean_bits, target_indices, group_mask, CODEBOOK)

    assert metrics.single_wer == 0.0
    assert metrics.single_ber == 0.0
    assert metrics.group_wer == 0.0
    assert metrics.group_ber == 0.0


def test_compute_metrics_ignores_masked_out_observations():
    target = 5
    good = _confident_logits_for(target, j=2)
    garbage = torch.zeros(2, NEO_20_10.n)  # would corrupt single-obs metrics if counted
    logits = torch.cat([good, garbage], dim=0).unsqueeze(0)
    clean_bits = CODEBOOK[target].unsqueeze(0)
    target_indices = torch.tensor([target])
    group_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    metrics = compute_metrics(logits, clean_bits, target_indices, group_mask, CODEBOOK)

    assert metrics.single_wer == 0.0
    assert metrics.group_wer == 0.0


def test_oracle_metrics_zero_noise_channel_is_always_correct():
    rng = torch.Generator().manual_seed(0)
    targets = torch.randint(0, 1024, (4,), generator=rng)
    clean_bits = CODEBOOK[targets]
    corrupted = clean_bits.unsqueeze(1).repeat(1, 3, 1).float()  # no corruption at all
    group_mask = torch.ones(4, 3)

    metrics = oracle_metrics(corrupted, clean_bits, targets, group_mask, CODEBOOK, channel=ZChannel(p=0.0))

    assert metrics.group_wer == 0.0
    assert metrics.group_ber == 0.0


def test_oracle_metrics_improves_with_more_repeats_under_noise():
    torch.manual_seed(0)
    target = 123
    clean_bits = CODEBOOK[target].unsqueeze(0)
    target_indices = torch.tensor([target])
    channel = ZChannel(p=0.4)

    def corrupt_group(j: int) -> torch.Tensor:
        clean_np = CODEBOOK[target].numpy()
        import numpy as np

        rng = np.random.default_rng(1)
        obs = [channel.corrupt(clean_np, rng) for _ in range(j)]
        return torch.as_tensor(np.stack(obs), dtype=torch.float32)

    few = corrupt_group(1).unsqueeze(0)
    many = corrupt_group(20).unsqueeze(0)

    few_metrics = oracle_metrics(few, clean_bits, target_indices, torch.ones(1, 1), CODEBOOK, channel=channel)
    many_metrics = oracle_metrics(many, clean_bits, target_indices, torch.ones(1, 20), CODEBOOK, channel=channel)

    assert many_metrics.group_ber <= few_metrics.group_ber
