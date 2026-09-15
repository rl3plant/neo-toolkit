"""WER/BER metrics and oracle sanity curves.

Ported from old_stuff/NEO_ECC/main.py's compute_group_metrics/
compute_decoded_wer/oracle_decodability_curve, rebuilt on decoding.py's
batched primitives instead of a Python for-loop calling bruteforce_map_decode
once per sample.

The oracle curve answers an explicit question: when should WER actually
reach 0? oracle_group_wer() decodes directly from
the *true* corruption (bits at oracle_logit confidence, no detector in the
loop at all) -- it's the best any detector could possibly do against a given
channel and group size J, so it's the right reference line for "is the
trained detector's WER actually good, or just low because the task is easy
at this (channel, J)."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neo_toolkit.ecc.channel import Channel
from neo_toolkit.ecc.decoding import codeword_log_likelihoods, combine_group_log_likelihoods, decode_single


@dataclass
class Metrics:
    single_wer: float  # decoding from one observation at a time, no aggregation
    single_ber: float
    group_wer: float  # decoding after combining all observations in a group
    group_ber: float


def compute_metrics(
    logits: torch.Tensor,
    clean_bits: torch.Tensor,
    target_indices: torch.Tensor,
    group_mask: torch.Tensor,
    codebook: torch.Tensor,
    *,
    channel: Channel | None = None,
) -> Metrics:
    """logits: (B, J, n) per-observation bit logits. clean_bits: (B, n).
    target_indices: (B,) true codeword indices. group_mask: (B, J)."""
    b, j, n = logits.shape
    flat_logits = logits.reshape(b * j, n)
    flat_decoded = decode_single(flat_logits, codebook).reshape(b, j)
    flat_valid = group_mask > 0.5

    single_decoded_bits = codebook[flat_decoded]  # (B, J, n)
    single_bit_errors = (single_decoded_bits != clean_bits.unsqueeze(1)) & flat_valid.unsqueeze(-1)
    single_word_errors = (flat_decoded != target_indices.unsqueeze(1)) & flat_valid
    n_valid_obs = flat_valid.sum().clamp(min=1)
    single_wer = float(single_word_errors.sum() / n_valid_obs)
    single_ber = float(single_bit_errors.sum() / (n_valid_obs * n))

    log_likelihoods = codeword_log_likelihoods(logits, codebook, channel=channel)
    group_scores = combine_group_log_likelihoods(log_likelihoods, group_mask)
    group_decoded = group_scores.argmax(dim=-1)
    group_decoded_bits = codebook[group_decoded]
    group_word_errors = group_decoded != target_indices
    group_bit_errors = (group_decoded_bits != clean_bits).sum(dim=-1)
    group_wer = float(group_word_errors.float().mean())
    group_ber = float(group_bit_errors.float().mean() / n)

    return Metrics(single_wer=single_wer, single_ber=single_ber, group_wer=group_wer, group_ber=group_ber)


def oracle_metrics(
    corrupted_bits: torch.Tensor,
    clean_bits: torch.Tensor,
    target_indices: torch.Tensor,
    group_mask: torch.Tensor,
    codebook: torch.Tensor,
    *,
    channel: Channel | None,
    oracle_logit: float = 8.0,
) -> Metrics:
    """The best any detector could do: decode straight from the true
    corrupted bits (as maximally-confident logits), no detector involved.
    The right reference line for "is WER low because decoding worked, or
    because the (channel, group size) combination makes the task easy."
    """
    logits = (2.0 * corrupted_bits.float() - 1.0) * oracle_logit
    return compute_metrics(logits, clean_bits, target_indices, group_mask, codebook, channel=channel)
