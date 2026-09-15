"""Error correction: encode/decode a 4x5 protein-position grid through a
(20, 10) linear code, with an exact, differentiable decoding "circuit"
(see decoding.py for why exact codebook enumeration is the right circuit
at this code size).
"""

from neo_toolkit.ecc.backgrounds import Crop, build_background_bank, load_bank, save_bank, split_bank_by_source
from neo_toolkit.ecc.channel import BinarySymmetricChannel, Channel, ZChannel
from neo_toolkit.ecc.code import LinearCode, NEO_20_10
from neo_toolkit.ecc.decoding import (
    codebook_tensor,
    codeword_log_likelihoods,
    decode_grouped,
    decode_single,
    grouped_nll_loss,
    nll_loss,
)
from neo_toolkit.ecc.detector import Detector
from neo_toolkit.ecc.metrics import Metrics, compute_metrics, oracle_metrics
from neo_toolkit.ecc.synthetic import Dataset, SyntheticConfig, generate_dataset, render_blobs
from neo_toolkit.ecc.training import TrainingConfig, TrainingHistory, train_detector
from neo_toolkit.ecc.yolo_dataset import load_categories_as_crops, load_category_as_crops, parse_yolo_seg_file

__all__ = [
    "BinarySymmetricChannel",
    "Channel",
    "Crop",
    "Dataset",
    "Detector",
    "LinearCode",
    "Metrics",
    "NEO_20_10",
    "SyntheticConfig",
    "TrainingConfig",
    "TrainingHistory",
    "ZChannel",
    "build_background_bank",
    "codebook_tensor",
    "codeword_log_likelihoods",
    "compute_metrics",
    "decode_grouped",
    "decode_single",
    "generate_dataset",
    "grouped_nll_loss",
    "load_bank",
    "load_categories_as_crops",
    "load_category_as_crops",
    "nll_loss",
    "oracle_metrics",
    "parse_yolo_seg_file",
    "render_blobs",
    "save_bank",
    "split_bank_by_source",
    "train_detector",
]
