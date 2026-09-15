"""Exact, differentiable decoding of the linear code: the "circuit" and its loss.

Everything here builds on one primitive, codeword_log_likelihoods(), which
scores all 2**k codewords at once against a batch of soft bit predictions.
That's the whole "circuit": with k=10 there are only 1024 codewords, so
exact posterior inference is a single dense tensor contraction, fully
batched (arbitrary leading dims) and fully differentiable through autograd --
no Python loop over the batch anywhere in this module.

This replaces old_stuff/NEO_ECC/neo.py's `main()` ("neurosymbolic layer" demo,
which recomputed everything for a single example, unbatched, once per
training step) and old_stuff/NEO_ECC/ecc.py's codeword_log_scores* functions
(which hand-special-cased rank-1 vs rank-2 vs rank-3 tensors instead of
using broadcasting), and old_stuff/NEO_ECC/main.py's bruteforce_map_decode
(called in a Python for-loop, once per sample) with a single batched call.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neo_toolkit.ecc.channel import Channel
from neo_toolkit.ecc.code import LinearCode


def codebook_tensor(code: LinearCode, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(code.codebook, dtype=dtype, device=device)


def bit_log_probs(bit_logits: torch.Tensor, channel: Channel | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """log P(bit=1), log P(bit=0) per position, any leading batch dims -> each (..., n).

    Shared by codeword_log_likelihoods (brute-force enumeration) and
    trellis.py (dynamic-programming enumeration) so both circuits score bits
    identically and only differ in how they combine those scores into
    per-codeword likelihoods.
    """
    if channel is None:
        return F.logsigmoid(bit_logits), F.logsigmoid(-bit_logits)
    q1 = torch.sigmoid(bit_logits)
    psi1, psi0 = channel.likelihood_transform(q1)
    eps = 1e-9
    return torch.log(psi1.clamp_min(eps)), torch.log(psi0.clamp_min(eps))


def codeword_log_likelihoods(
    bit_logits: torch.Tensor,
    codebook: torch.Tensor,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """Log-likelihood of every codeword given per-bit logits, any leading batch dims.

    bit_logits: (..., n) raw detector logits (pre-sigmoid).
    codebook:   (K, n) {0, 1} tensor of the K = 2**k valid codewords.
    channel:    if given, reinterpret sigmoid(bit_logits) as the channel's
                observed-bit probability rather than the true-bit probability
                (see channel.py); if None, bit_logits are treated as scoring
                the true bit directly.
    returns:    (..., K) log-likelihood of each codeword.
    """
    codebook = codebook.to(dtype=bit_logits.dtype, device=bit_logits.device)
    log_p1, log_p0 = bit_log_probs(bit_logits, channel)
    return torch.einsum("...n,kn->...k", log_p1, codebook) + torch.einsum("...n,kn->...k", log_p0, 1.0 - codebook)


def combine_group_log_likelihoods(log_likelihoods: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
    """Combine J conditionally-independent repeated observations by summing their
    log-likelihoods. log_likelihoods: (..., J, K), group_mask: (..., J) in {0, 1}."""
    return (log_likelihoods * group_mask.unsqueeze(-1)).sum(dim=-2)


def decode_single(bit_logits: torch.Tensor, codebook: torch.Tensor, *, channel: Channel | None = None) -> torch.Tensor:
    """Exact MAP decode from a single noisy observation. bit_logits: (..., n) -> (...,)."""
    return codeword_log_likelihoods(bit_logits, codebook, channel=channel).argmax(dim=-1)


def decode_grouped(
    bit_logits: torch.Tensor,
    codebook: torch.Tensor,
    group_mask: torch.Tensor,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """Exact MAP decode after combining J repeated observations per group.

    bit_logits: (..., J, n), group_mask: (..., J) -> (...,).
    """
    log_likelihoods = codeword_log_likelihoods(bit_logits, codebook, channel=channel)
    group_scores = combine_group_log_likelihoods(log_likelihoods, group_mask)
    return group_scores.argmax(dim=-1)


def nll_loss(
    bit_logits: torch.Tensor,
    target_indices: torch.Tensor,
    codebook: torch.Tensor,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """Exact -log P(true codeword | observed bits), averaged over the batch.

    bit_logits: (B, n), target_indices: (B,) codebook row indices.
    """
    log_likelihoods = codeword_log_likelihoods(bit_logits, codebook, channel=channel)
    return _nll_from_log_likelihoods(log_likelihoods, target_indices)


def grouped_nll_loss(
    bit_logits: torch.Tensor,
    target_indices: torch.Tensor,
    codebook: torch.Tensor,
    group_mask: torch.Tensor,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """Exact grouped -log P(true codeword | J observations), averaged over the batch.

    bit_logits: (B, J, n), target_indices: (B,), group_mask: (B, J).
    """
    log_likelihoods = codeword_log_likelihoods(bit_logits, codebook, channel=channel)
    group_scores = combine_group_log_likelihoods(log_likelihoods, group_mask)
    return _nll_from_log_likelihoods(group_scores, target_indices)


def _nll_from_log_likelihoods(scores: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(scores, dim=-1)
    true_scores = scores.gather(-1, target_indices.unsqueeze(-1)).squeeze(-1)
    return (log_z - true_scores).mean()
