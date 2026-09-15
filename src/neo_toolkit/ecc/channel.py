"""Channel models: how bits get corrupted, and how a decoder should weigh
a detector's soft prediction knowing which channel produced it.

A channel is used two ways: to corrupt clean bits when simulating training
data (corrupt()), and to correctly reweigh a detector's soft bit probability
when decoding (likelihood_transform()). The two need to agree, e.g. under a
Z-channel a detector reporting "probably 0" is ambiguous (a genuine 0, or a
corrupted 1), while "probably 1" is unambiguous (a 1 can only survive
uncorrupted) -- treating both the same, as a plain sigmoid, throws that
asymmetry away. Ported from old_stuff/NEO_ECC/data_pipeline.py's corrupt_bits
and ecc.py's channel_type branching, unified into one place per channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


class Channel(Protocol):
    def corrupt(self, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    def likelihood_transform(self, q1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Given the detector's P(bit=1), return (psi1, psi0): the likelihood
        weight to use when scoring a candidate codeword bit of 1 or 0."""
        ...


@dataclass(frozen=True)
class ZChannel:
    """Protein dropout: a 1 can flip to 0 with probability p; a 0 never flips."""

    p: float

    def corrupt(self, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        bits = bits.astype(np.uint8)
        drop = (rng.random(bits.shape) < self.p) & (bits == 1)
        return np.where(drop, 0, bits).astype(np.uint8)

    def likelihood_transform(self, q1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi1 = (1.0 - self.p) * q1 + self.p * (1.0 - q1)
        psi0 = 1.0 - q1
        return psi1, psi0


@dataclass(frozen=True)
class BinarySymmetricChannel:
    """Random bit flips in either direction, each with probability p."""

    p: float

    def corrupt(self, bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        flips = (rng.random(bits.shape) < self.p).astype(np.uint8)
        return np.bitwise_xor(bits.astype(np.uint8), flips)

    def likelihood_transform(self, q1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi1 = (1.0 - self.p) * q1 + self.p * (1.0 - q1)
        psi0 = self.p * q1 + (1.0 - self.p) * (1.0 - q1)
        return psi1, psi0
