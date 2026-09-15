import numpy as np

from neo_toolkit.ecc import gf2


def test_matmul_reduces_mod_2():
    a = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([[1, 0], [1, 1]], dtype=np.uint8)

    result = gf2.matmul(a, b)

    assert result.dtype == np.uint8
    assert np.array_equal(result, np.array([[0, 1], [1, 1]], dtype=np.uint8))


def test_rank_of_identity_is_full():
    identity = np.eye(5, dtype=np.uint8)

    assert gf2.rank(identity) == 5


def test_rank_detects_linear_dependence():
    matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)  # row2 = row0 XOR row1

    assert gf2.rank(matrix) == 2


def test_rank_of_zero_matrix_is_zero():
    assert gf2.rank(np.zeros((4, 4), dtype=np.uint8)) == 0
