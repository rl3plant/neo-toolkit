"""A compiled trellis/circuit alternative to decoding.py's brute-force enumeration.

decoding.py's docstring already explains why brute force was chosen: with
k=10 there are only 2**10=1024 codewords, so scoring all of them at once is
one dense tensor contraction. This module builds the other kind of exact
circuit -- a BCJR-style forward/backward trellis over NEO_20_10.H, walked one
bit position at a time -- to answer two questions empirically rather than by
assumption: does it compute the same thing, and is it actually cheaper.

It is NOT cheaper for this code. A trellis's state space at bit-boundary i is
{0,1}^|A_i|, where A_i is the set of parity rows whose support spans that
boundary; the worst-case width is bounded by 2**min(i, n-i, r). Trellises pay
off when a code's parity-check matrix is sparse/structured (convolutional and
LDPC codes are designed for this), so most rows only span a few boundaries
and the trellis narrows to a fat middle and thin ends. NEO_20_10.H's P
submatrix is dense (chosen for minimum distance, not trellis-friendliness):
every row's active-set membership was checked with GF(2) rank (not just
support-column counting, in case some rows turned out to be redundant given
others already active -- they aren't, H has full row rank 10) and the active
state space hits the full 2**10=1024 states at the midpoint, identical to
brute force, and the *sum* of state-transition work across all 20 boundaries
(3070) is larger than one brute-force pass. So this module exists purely for
cross-validation (does a differently-derived circuit agree with the
brute-force one, and does training through it change anything) -- not
because it's expected to help.

Construction (build_trellis): for each row j of H, its "span" is
[start_j, end_j], the first and last nonzero column. At boundary i (after
processing bits 0..i-1), the active rows are {j : start_j < i <= end_j}; the
trellis state there is the vector of each active row's partial XOR-sum over
the bits it's seen so far. Walking from boundary i to i+1 by observing bit
x_i: a row with start_j == i "starts" (its partial sum initializes to
H[j,i]*x_i); a continuing row XORs in H[j,i]*x_i; a row with end_j == i
"ends" -- its parity check must land on exactly 0 or the path is pruned. All
of this is precomputed once per (row-order, code) as a flat transition table
per boundary; the actual forward/backward passes below are batched, fully
differentiable PyTorch ops over that fixed table, no Python loop over states
or over the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neo_toolkit.ecc.channel import Channel
from neo_toolkit.ecc.code import LinearCode
from neo_toolkit.ecc.decoding import bit_log_probs


@dataclass(frozen=True)
class TrellisStep:
    """Boundary i -> boundary i+1, observing bit x_i.

    n_in/n_out are *state counts* (2**|active rows|), not row counts.
    transition[2*old_state + bit] = new_state, or n_out (a trash bucket) if
    observing that bit from that state violates an ending row's parity check.
    """

    n_in: int
    n_out: int
    transition: np.ndarray  # (n_in * 2,) int64, values in [0, n_out]


@dataclass(frozen=True)
class Trellis:
    code: LinearCode
    steps: tuple[TrellisStep, ...]  # length code.n


def _row_spans(H: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    for row in H:
        nz = np.nonzero(row)[0]
        spans.append((int(nz.min()), int(nz.max())))
    return spans


def _active_rows(spans: list[tuple[int, int]], boundary: int) -> tuple[int, ...]:
    return tuple(j for j, (start, end) in enumerate(spans) if start < boundary <= end)


def build_trellis(code: LinearCode) -> Trellis:
    H = code.H
    n = code.n
    spans = _row_spans(H)

    steps = []
    for i in range(n):
        a_in = _active_rows(spans, i)
        a_out = _active_rows(spans, i + 1)
        n_in, n_out = 2 ** len(a_in), 2 ** len(a_out)
        pos_in = {row: p for p, row in enumerate(a_in)}
        pos_out = {row: p for p, row in enumerate(a_out)}
        starting = [row for row in a_out if row not in pos_in]
        ending = [row for row in a_in if row not in pos_out]
        continuing = [row for row in a_in if row in pos_out]

        transition = np.full(n_in * 2, n_out, dtype=np.int64)  # default: pruned
        for old_state in range(n_in):
            old_bits = {row: (old_state >> pos_in[row]) & 1 for row in a_in}
            for bit in (0, 1):
                ok = True
                new_bits: dict[int, int] = {}
                for row in continuing:
                    new_bits[row] = old_bits[row] ^ (int(H[row, i]) & bit)
                for row in ending:
                    if old_bits[row] ^ (int(H[row, i]) & bit) != 0:
                        ok = False
                        break
                if not ok:
                    continue
                for row in starting:
                    new_bits[row] = int(H[row, i]) & bit
                new_state = 0
                for row, pos in pos_out.items():
                    new_state |= new_bits[row] << pos
                transition[old_state * 2 + bit] = new_state
        steps.append(TrellisStep(n_in=n_in, n_out=n_out, transition=transition))

    return Trellis(code=code, steps=tuple(steps))


def _step_logp_pair(log_p1: torch.Tensor, log_p0: torch.Tensor, i: int) -> torch.Tensor:
    return torch.stack([log_p0[..., i], log_p1[..., i]], dim=-1)  # (..., 2), index by bit value


def trellis_log_z(bit_logits: torch.Tensor, trellis: Trellis, *, channel: Channel | None = None) -> torch.Tensor:
    """log-partition-function: logsumexp over every codeword's log-likelihood.

    Equivalent to codeword_log_likelihoods(bit_logits, codebook, channel=channel)
    .logsumexp(-1) -- see tests/ecc/test_trellis.py for the equivalence check.
    bit_logits: (..., n) -> (...,).
    """
    log_p1, log_p0 = bit_log_probs(bit_logits, channel)
    return _trellis_log_z_from_logp(log_p1, log_p0, trellis)


def trellis_decode_bits(bit_logits: torch.Tensor, trellis: Trellis, *, channel: Channel | None = None) -> torch.Tensor:
    """Exact MAP codeword (as bits, not a codebook index) via max-product + backtrace.

    bit_logits: (..., n) -> (..., n) {0, 1} float tensor. Not meant to be
    differentiated through (argmax/backtrace); used for evaluation only, same
    as decoding.decode_single.
    """
    log_p1, log_p0 = bit_log_probs(bit_logits, channel)
    batch_shape = log_p1.shape[:-1]
    device = log_p1.device
    alpha = torch.zeros(*batch_shape, 1, dtype=log_p1.dtype, device=device)
    backptrs: list[torch.Tensor] = []  # per step: (..., n_out) source index into [0, n_in_states*2)

    for i, step in enumerate(trellis.steps):
        logp_i = _step_logp_pair(log_p1, log_p0, i)
        val = alpha.unsqueeze(-1) + logp_i.unsqueeze(-2)
        val_flat = val.reshape(*batch_shape, -1)
        n_sources = val_flat.shape[-1]
        target = torch.as_tensor(step.transition, device=device)
        target = target.expand(*batch_shape, n_sources)

        acc = torch.full((*batch_shape, step.n_out + 1), float("-inf"), dtype=val_flat.dtype, device=device)
        acc = acc.scatter_reduce(-1, target, val_flat, reduce="amax", include_self=True)

        winning_value = acc.gather(-1, target)  # (..., n_sources): value of the bucket each source landed in
        is_argmax = val_flat == winning_value
        source_idx = torch.arange(n_sources, device=device).expand(*batch_shape, n_sources)
        sentinel = n_sources
        candidate = torch.where(is_argmax, source_idx, sentinel)
        winner = torch.full((*batch_shape, step.n_out + 1), sentinel, dtype=torch.long, device=device)
        winner = winner.scatter_reduce(-1, target, candidate, reduce="amin", include_self=True)

        backptrs.append(winner[..., : step.n_out])
        alpha = acc[..., : step.n_out]

    n = len(trellis.steps)
    bits = torch.zeros(*batch_shape, n, dtype=torch.float32, device=device)
    state = torch.zeros(batch_shape, dtype=torch.long, device=device)  # final boundary has exactly 1 state: 0
    for i in range(n - 1, -1, -1):
        source = backptrs[i].gather(-1, state.unsqueeze(-1)).squeeze(-1)
        bits[..., i] = (source % 2).float()
        state = source // 2

    return bits


def trellis_nll_loss(
    bit_logits: torch.Tensor,
    target_indices: torch.Tensor,
    codebook: torch.Tensor,
    trellis: Trellis,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """-log P(true codeword | observed bits) via the trellis's log Z, averaged over the batch.

    The true-codeword score itself needs no trellis: it's a direct dot
    product with the (known) target bits, same value the brute-force circuit
    would compute for that one codeword. bit_logits: (B, n), target_indices: (B,).
    """
    log_z = trellis_log_z(bit_logits, trellis, channel=channel)
    log_p1, log_p0 = bit_log_probs(bit_logits, channel)
    target_bits = codebook.to(bit_logits.device)[target_indices]
    true_score = (target_bits * log_p1 + (1.0 - target_bits) * log_p0).sum(dim=-1)
    return (log_z - true_score).mean()


def trellis_grouped_nll_loss(
    bit_logits: torch.Tensor,
    target_indices: torch.Tensor,
    codebook: torch.Tensor,
    group_mask: torch.Tensor,
    trellis: Trellis,
    *,
    channel: Channel | None = None,
) -> torch.Tensor:
    """Exact grouped -log P(true codeword | J observations) via the trellis.

    A group's log-likelihood for any fixed codeword is a per-bit-position sum
    over its J observations (see decoding.combine_group_log_likelihoods'
    docstring reasoning), so J observations can be combined into one set of
    per-bit log-probabilities *before* the trellis pass, rather than running
    the trellis J times and summing -- both give the same answer.
    bit_logits: (B, J, n), target_indices: (B,), group_mask: (B, J).
    """
    log_p1, log_p0 = bit_log_probs(bit_logits, channel)
    mask = group_mask.unsqueeze(-1)
    combined_log_p1 = (log_p1 * mask).sum(dim=-2)  # (B, n)
    combined_log_p0 = (log_p0 * mask).sum(dim=-2)

    log_z = _trellis_log_z_from_logp(combined_log_p1, combined_log_p0, trellis)
    target_bits = codebook.to(bit_logits.device)[target_indices]
    true_score = (target_bits * combined_log_p1 + (1.0 - target_bits) * combined_log_p0).sum(dim=-1)
    return (log_z - true_score).mean()


def _trellis_log_z_from_logp(log_p1: torch.Tensor, log_p0: torch.Tensor, trellis: Trellis) -> torch.Tensor:
    """Same forward pass as trellis_log_z, but from already-combined log-probs."""
    batch_shape = log_p1.shape[:-1]
    alpha = torch.zeros(*batch_shape, 1, dtype=log_p1.dtype, device=log_p1.device)
    for i, step in enumerate(trellis.steps):
        logp_i = _step_logp_pair(log_p1, log_p0, i)
        val = alpha.unsqueeze(-1) + logp_i.unsqueeze(-2)
        val_flat = val.reshape(*batch_shape, -1)
        target = torch.as_tensor(step.transition, device=val_flat.device)
        target = target.expand(*batch_shape, target.shape[-1])
        shift = val_flat.max(dim=-1, keepdim=True).values
        exp_val = (val_flat - shift).exp()
        acc = torch.zeros(*batch_shape, step.n_out + 1, dtype=val_flat.dtype, device=val_flat.device)
        acc = acc.scatter_add(-1, target, exp_val)
        alpha = shift + torch.log(acc[..., : step.n_out].clamp_min(1e-38))
    return alpha.squeeze(-1)
