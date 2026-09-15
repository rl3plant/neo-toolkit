"""Minimal GF(2) (mod-2) linear algebra: matmul, row-reduction, rank.

Kept separate from code.py because it's pure linear algebra with no
domain meaning. The original (old_stuff/NEO_ECC/neo.py) did GF(2) arithmetic
as float64 torch matmuls followed by `% 2`, which works but obscures that
this is bit arithmetic; here it's plain numpy uint8 + XOR row operations.
"""

from __future__ import annotations

import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix product over GF(2): ordinary matmul, reduced mod 2."""
    return ((a.astype(np.int64) @ b.astype(np.int64)) % 2).astype(np.uint8)


def row_reduce(matrix: np.ndarray) -> np.ndarray:
    """Reduced row-echelon form of `matrix` over GF(2), via XOR row ops."""
    m = (matrix.astype(np.uint8) % 2).copy()
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        pivot = next((r for r in range(pivot_row, rows) if m[r, col]), None)
        if pivot is None:
            continue
        m[[pivot_row, pivot]] = m[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and m[r, col]:
                m[r] ^= m[pivot_row]
        pivot_row += 1
        if pivot_row == rows:
            break
    return m


def rank(matrix: np.ndarray) -> int:
    """Rank of `matrix` over GF(2)."""
    reduced = row_reduce(matrix)
    return int(np.any(reduced, axis=1).sum())
