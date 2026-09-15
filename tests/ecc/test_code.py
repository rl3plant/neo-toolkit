import numpy as np
import pytest

from neo_toolkit.ecc.code import LinearCode, NEO_20_10, build_codebook, from_systematic_parity, verify


def test_neo_20_10_shape():
    assert NEO_20_10.k == 10
    assert NEO_20_10.n == 20
    assert NEO_20_10.codebook.shape == (1024, 20)


def test_neo_20_10_is_well_formed():
    verify(NEO_20_10)  # raises AssertionError if not


def test_neo_20_10_minimum_distance_is_6():
    assert NEO_20_10.minimum_distance() == 6


def test_message_zero_encodes_to_all_zero_codeword():
    assert not np.any(NEO_20_10.codebook[0])


def test_encode_matches_codebook_for_random_messages():
    rng = np.random.default_rng(0)
    messages = rng.integers(0, 2, size=(50, NEO_20_10.k), dtype=np.uint8)

    encoded = NEO_20_10.encode(messages)

    assert np.all(NEO_20_10.is_codeword(encoded))
    for message, codeword in zip(messages, encoded):
        index = int("".join(map(str, message)), 2)
        assert np.array_equal(codeword, NEO_20_10.codebook[index])


def test_is_codeword_rejects_a_bit_flip():
    codeword = NEO_20_10.codebook[5].copy()
    codeword[0] ^= 1

    assert not NEO_20_10.is_codeword(codeword)


def test_verify_rejects_a_rank_deficient_generator():
    # G = [I_k | P] is always full rank regardless of P (the identity block
    # alone guarantees it), so a rank-deficient G has to be built directly.
    g = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)  # rows 0 and 1 duplicate
    broken = LinearCode(G=g, H=np.zeros((0, 3), dtype=np.uint8), codebook=build_codebook(g))

    with pytest.raises(AssertionError):
        verify(broken)


def test_build_codebook_matches_brute_force_for_small_code():
    p = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    code = from_systematic_parity(p)

    expected = []
    for m in range(4):
        bits = np.array([(m >> 1) & 1, m & 1], dtype=np.uint8)
        expected.append((bits @ code.G) % 2)

    assert np.array_equal(build_codebook(code.G), np.array(expected, dtype=np.uint8))
