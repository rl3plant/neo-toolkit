import torch

from neo_toolkit.ecc.detector import Detector


def test_forward_output_shape():
    detector = Detector(in_channels=3, base_channels=8, n_layers=2)
    x = torch.zeros(4, 3, 64, 64)

    logits = detector(x)

    assert logits.shape == (4, 20)


def test_n_layers_four_is_valid_and_produces_correct_shape():
    detector = Detector(in_channels=3, base_channels=8, n_layers=4)
    x = torch.zeros(2, 3, 128, 128)

    logits = detector(x)

    assert logits.shape == (2, 20)


def test_invalid_n_layers_raises():
    import pytest

    with pytest.raises(ValueError):
        Detector(n_layers=5)


def test_batch_norm_produces_correct_shape_and_survives_eval_with_batch_size_one():
    detector = Detector(base_channels=8, n_layers=2, norm="batch")
    detector.train()
    detector(torch.rand(4, 3, 64, 64))  # a training step so BatchNorm has running stats
    detector.eval()

    with torch.no_grad():
        logits = detector(torch.rand(1, 3, 64, 64))  # batch size 1, as end_to_end_demo.py uses

    assert logits.shape == (1, 20)


def test_group_norm_produces_correct_shape():
    detector = Detector(base_channels=8, n_layers=2, norm="group")
    x = torch.rand(4, 3, 64, 64)

    logits = detector(x)

    assert logits.shape == (4, 20)


def test_dropout_produces_correct_shape():
    detector = Detector(base_channels=8, n_layers=2, dropout=0.3)
    x = torch.rand(4, 3, 64, 64)

    logits = detector(x)

    assert logits.shape == (4, 20)


def test_kernel_size_five_produces_correct_shape():
    detector = Detector(base_channels=8, n_layers=2, kernel_size=5)
    x = torch.rand(4, 3, 64, 64)

    logits = detector(x)

    assert logits.shape == (4, 20)


def test_invalid_norm_raises():
    import pytest

    with pytest.raises(ValueError):
        Detector(norm="layer")


def test_even_kernel_size_raises():
    import pytest

    with pytest.raises(ValueError):
        Detector(kernel_size=4)
