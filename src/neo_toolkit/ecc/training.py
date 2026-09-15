"""Train a Detector against one of three learning objectives.

Ported from old_stuff/NEO_ECC/main.py's train_variants() and the three
SINGLE_VARIANTS it compared (det_cp / det_cp_ecc_c / ecc_c_only), rebuilt on
decoding.py's batched primitives. The three objectives, matching D3.7 and
the NEO_ECC "report plan" notes:

- "supervised" ("traditional"): masked BCE against the observed (corrupted)
  bits directly -- the detector never sees the code at all.
- "neurosymbolic": only the exact decoding NLL loss against the true
  codeword (decoding.grouped_nll_loss) -- the detector is trained purely to
  make the code's constraints pick out the right codeword.
- "hybrid": both losses, summed with a weight on the NLL term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from neo_toolkit.ecc.channel import Channel
from neo_toolkit.ecc.decoding import grouped_nll_loss
from neo_toolkit.ecc.detector import Detector
from neo_toolkit.ecc.metrics import Metrics, compute_metrics
from neo_toolkit.ecc.synthetic import Dataset
from neo_toolkit.ecc.trellis import Trellis, trellis_grouped_nll_loss

Objective = Literal["supervised", "neurosymbolic", "hybrid"]
Circuit = Literal["brute_force", "trellis"]


@dataclass
class TrainingConfig:
    objective: Objective = "hybrid"
    epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    lambda_ecc: float = 0.05
    channel: Channel | None = None  # None -> the ecc loss scores logits directly, not through a channel model
    device: str = "cpu"
    # Which decoding circuit the ecc loss differentiates through -- see
    # trellis.py for why "trellis" exists (cross-validation, not speed).
    circuit: Circuit = "brute_force"
    trellis: Trellis | None = None  # required when circuit == "trellis"


@dataclass
class TrainingHistory:
    epoch_loss: list[float] = field(default_factory=list)
    val_metrics: list[Metrics] = field(default_factory=list)
    best_epoch: int = field(default=-1)
    best_state_dict: dict = field(default=None, repr=False)


def _masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    denom = (mask.unsqueeze(-1).sum() * logits.size(-1)).clamp(min=1.0)
    return (loss * mask.unsqueeze(-1)).sum() / denom


def _forward_logits(detector: Detector, images_bjhwc: torch.Tensor, *, max_forward_batch: int = 64) -> torch.Tensor:
    """Flatten (batch, J) into one dimension for a single conv forward pass,
    chunked to bound peak memory. batch_size x J can reach several hundred
    images (e.g. 32 x 20), and combined with a wide/shallow architecture
    (large feature maps, few downsampling layers) that's enough to OOM a GPU
    in one allocation -- chunking keeps memory bounded regardless of
    (batch_size, J, architecture). For stateless layers (conv/relu/pool,
    dropout in eval mode) this changes nothing about the gradients computed,
    just the number of kernel launches; for BatchNorm specifically, each
    chunk gets its own batch statistics rather than one shared across all
    b*j images -- a minor deviation (chunk size 64 is already a normal
    single-batch size for BatchNorm), not a correctness bug.
    """
    b, j, h, w, c = images_bjhwc.shape
    flat = images_bjhwc.reshape(b * j, h, w, c).permute(0, 3, 1, 2)
    if flat.shape[0] <= max_forward_batch:
        logits = detector(flat)
    else:
        chunks = [detector(flat[i : i + max_forward_batch]) for i in range(0, flat.shape[0], max_forward_batch)]
        logits = torch.cat(chunks, dim=0)
    return logits.reshape(b, j, -1)


def _step_loss(
    logits: torch.Tensor,
    corrupted_bits: torch.Tensor,
    target_indices: torch.Tensor,
    group_mask: torch.Tensor,
    codebook: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    if cfg.objective == "supervised":
        return _masked_bce_loss(logits, corrupted_bits, group_mask)
    if cfg.objective == "neurosymbolic":
        return _ecc_loss(logits, target_indices, codebook, group_mask, cfg)
    if cfg.objective == "hybrid":
        det_loss = _masked_bce_loss(logits, corrupted_bits, group_mask)
        ecc_loss = _ecc_loss(logits, target_indices, codebook, group_mask, cfg)
        return det_loss + cfg.lambda_ecc * ecc_loss
    raise ValueError(f"unknown objective {cfg.objective!r}")


def _ecc_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    codebook: torch.Tensor,
    group_mask: torch.Tensor,
    cfg: TrainingConfig,
) -> torch.Tensor:
    if cfg.circuit == "brute_force":
        return grouped_nll_loss(logits, target_indices, codebook, group_mask, channel=cfg.channel)
    if cfg.circuit == "trellis":
        if cfg.trellis is None:
            raise ValueError("TrainingConfig.trellis must be set when circuit='trellis'")
        return trellis_grouped_nll_loss(logits, target_indices, codebook, group_mask, cfg.trellis, channel=cfg.channel)
    raise ValueError(f"unknown circuit {cfg.circuit!r}")


def _to_tensors(dataset: Dataset, codebook: torch.Tensor, device: str):
    return (
        torch.as_tensor(dataset.images, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.clean_bits, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.corrupted_bits, dtype=torch.float32, device=device),
        torch.as_tensor(dataset.message_indices, dtype=torch.long, device=device),
        torch.as_tensor(dataset.group_mask, dtype=torch.float32, device=device),
    )


def train_detector(
    detector: Detector,
    train_dataset: Dataset,
    val_dataset: Dataset,
    codebook: torch.Tensor,
    cfg: TrainingConfig,
) -> TrainingHistory:
    detector.to(cfg.device)
    codebook = codebook.to(cfg.device)
    optimizer = torch.optim.Adam(detector.parameters(), lr=cfg.lr)

    train_images, _, train_corrupted, train_targets, train_mask = _to_tensors(train_dataset, codebook, cfg.device)
    val_images, val_clean, val_corrupted, val_targets, val_mask = _to_tensors(val_dataset, codebook, cfg.device)

    n_train = train_images.shape[0]
    history = TrainingHistory()

    for _epoch in range(cfg.epochs):
        detector.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            logits = _forward_logits(detector, train_images[idx])
            loss = _step_loss(logits, train_corrupted[idx], train_targets[idx], train_mask[idx], codebook, cfg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        history.epoch_loss.append(total_loss / max(1, n_batches))

        detector.eval()
        with torch.no_grad():
            val_logits = _forward_logits(detector, val_images)
            metrics = compute_metrics(val_logits, val_clean, val_targets, val_mask, codebook, channel=cfg.channel)
            history.val_metrics.append(metrics)
            if history.best_state_dict is None or metrics.group_wer < history.val_metrics[history.best_epoch].group_wer:
                history.best_epoch = len(history.val_metrics) - 1
                history.best_state_dict = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}

    return history
