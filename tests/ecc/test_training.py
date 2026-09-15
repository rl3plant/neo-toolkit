import numpy as np
import pytest
import torch

from neo_toolkit.ecc.backgrounds import Crop
from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.decoding import codebook_tensor
from neo_toolkit.ecc.detector import Detector
from neo_toolkit.ecc.synthetic import SyntheticConfig, generate_dataset
from neo_toolkit.ecc.trellis import build_trellis
from neo_toolkit.ecc.training import TrainingConfig, _forward_logits, train_detector

CODEBOOK = codebook_tensor(NEO_20_10)
TRELLIS = build_trellis(NEO_20_10)


def _fake_bank(n: int = 4, size: int = 16) -> list[Crop]:
    rng = np.random.default_rng(0)
    bank = []
    for i in range(n):
        mask = np.zeros((size, size), dtype=np.float32)
        mask[2 : size - 2, 2 : size - 2] = 1.0
        image = rng.uniform(0.0, 0.3, size=(size, size, 3)).astype(np.float32)
        bank.append(Crop(image=image, mask=mask, source=f"fake_{i}.png"))
    return bank


def _tiny_datasets():
    bank = _fake_bank()
    train_cfg = SyntheticConfig(num_messages=8, group_j_min=2, group_j_max=2, channel=ZChannel(p=0.2), seed=0)
    val_cfg = SyntheticConfig(num_messages=4, group_j_min=2, group_j_max=2, channel=ZChannel(p=0.2), seed=1)
    return generate_dataset(bank, NEO_20_10, train_cfg), generate_dataset(bank, NEO_20_10, val_cfg)


@pytest.mark.parametrize("objective", ["supervised", "neurosymbolic", "hybrid"])
def test_train_detector_runs_and_produces_history_for_each_objective(objective):
    train_ds, val_ds = _tiny_datasets()
    detector = Detector(in_channels=3, base_channels=4, n_layers=1)
    cfg = TrainingConfig(objective=objective, epochs=2, batch_size=4, lr=1e-2)

    history = train_detector(detector, train_ds, val_ds, CODEBOOK, cfg)

    assert len(history.epoch_loss) == 2
    assert len(history.val_metrics) == 2
    assert all(np.isfinite(loss) for loss in history.epoch_loss)


@pytest.mark.parametrize("objective", ["neurosymbolic", "hybrid"])
def test_train_detector_trellis_circuit_runs_and_tracks_brute_force_loss(objective):
    # Not just "does it run": the two circuits compute the same loss (see
    # test_trellis.py), so training the same initial weights on the same
    # batches through either circuit should produce near-identical loss
    # trajectories, not just similarly-shaped ones.
    train_ds, val_ds = _tiny_datasets()

    torch.manual_seed(0)
    detector_bf = Detector(in_channels=3, base_channels=4, n_layers=1)
    cfg_bf = TrainingConfig(objective=objective, epochs=3, batch_size=4, lr=1e-2, circuit="brute_force")
    history_bf = train_detector(detector_bf, train_ds, val_ds, CODEBOOK, cfg_bf)

    torch.manual_seed(0)
    detector_tr = Detector(in_channels=3, base_channels=4, n_layers=1)
    cfg_tr = TrainingConfig(objective=objective, epochs=3, batch_size=4, lr=1e-2, circuit="trellis", trellis=TRELLIS)
    history_tr = train_detector(detector_tr, train_ds, val_ds, CODEBOOK, cfg_tr)

    assert history_tr.epoch_loss == pytest.approx(history_bf.epoch_loss, abs=1e-4)


def test_train_detector_trellis_circuit_requires_trellis_argument():
    train_ds, val_ds = _tiny_datasets()
    detector = Detector(in_channels=3, base_channels=4, n_layers=1)
    cfg = TrainingConfig(objective="neurosymbolic", epochs=1, batch_size=4, lr=1e-2, circuit="trellis")

    with pytest.raises(ValueError, match="trellis"):
        train_detector(detector, train_ds, val_ds, CODEBOOK, cfg)


def test_train_detector_neurosymbolic_loss_decreases_when_overfitting_a_tiny_set():
    bank = _fake_bank()
    cfg = SyntheticConfig(num_messages=4, group_j_min=4, group_j_max=4, channel=ZChannel(p=0.0), seed=0)
    dataset = generate_dataset(bank, NEO_20_10, cfg)
    detector = Detector(in_channels=3, base_channels=8, n_layers=1)
    train_cfg = TrainingConfig(objective="neurosymbolic", epochs=15, batch_size=4, lr=1e-2)

    history = train_detector(detector, dataset, dataset, CODEBOOK, train_cfg)

    assert history.epoch_loss[-1] < history.epoch_loss[0]


def test_forward_logits_chunking_matches_unchunked_for_stateless_architecture():
    # No BatchNorm -> chunking must be purely a memory optimization, not
    # change the result at all.
    detector = Detector(in_channels=3, base_channels=4, n_layers=1, norm="none")
    detector.eval()
    images = torch.rand(3, 5, 16, 16, 3)  # b=3, j=5 -> 15 images, forces >1 chunk at max_forward_batch=4

    with torch.no_grad():
        chunked = _forward_logits(detector, images, max_forward_batch=4)
        unchunked = _forward_logits(detector, images, max_forward_batch=1000)

    assert torch.allclose(chunked, unchunked, atol=1e-6)
    assert chunked.shape == (3, 5, 20)


def test_forward_logits_chunking_bounds_peak_batch_size():
    calls = []
    detector = Detector(in_channels=3, base_channels=4, n_layers=1)
    original_forward = detector.forward

    def spy_forward(x):
        calls.append(x.shape[0])
        return original_forward(x)

    detector.forward = spy_forward
    images = torch.rand(2, 10, 16, 16, 3)  # 20 images total

    _forward_logits(detector, images, max_forward_batch=6)

    assert all(c <= 6 for c in calls)
    assert sum(calls) == 20
