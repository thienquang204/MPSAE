"""Supervised linear probe on the frozen sparse code.

This is classification, not the retrieval metric, and it answers a specific
question about the baselines: how much class information survives the Top-K
gate. The protocol follows the CSR author's answer in issue #18 of the CSR
repository -- a single ``nn.Linear`` over the **full hidden-dimensional** latent
trained with cross-entropy. Probing the compact ``k`` values instead would
measure a different object, because the latent's identity (*which* dictionary
atom fired) is most of the signal.

The encoder is never updated. Only training labels fit the probe; the validation
labels score it once, at the final epoch, with no early stopping and no
selection, so the number is not optimistically biased.

It runs once per (backbone, method, K, seed) -- 690 cells in the plan's E1 grid
alone -- so it is deliberately cheap: ten epochs of one matrix multiply per
batch, with the codes densified on the GPU a batch at a time.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..logs import Progress, format_duration, get_logger
from .encode import CodeSource
from .knn import fp32_matmul

PROTOCOL = "csr-issue18-full-latent-linear-probe"


def linear_probe(train_codes: Any, train_labels: Any, val_codes: Any, val_labels: Any, *,
                 hidden_dim: int, device: torch.device | str, epochs: int = 10,
                 batch_size: int = 4096, lr: float = 1e-3, weight_decay: float = 0.0,
                 seed: int = 1227, num_classes: int = 1000,
                 progress_stage: str = "probe") -> dict[str, Any]:
    """Fit one softmax classifier on the full latent and score it once.

    ``train_codes``/``val_codes`` are compact ``(indices, values)`` codes as
    produced by :func:`~mpsae.evaluate.encode.encode_split` (dense matrices are
    accepted too, for the dense reference probe); each batch is expanded to
    ``hidden_dim`` on the GPU.

    Determinism: the head is zero-initialised, so it consumes no RNG and is
    identical across K and across methods; batch order comes from a NumPy
    generator seeded by ``(seed, epoch)``, so a rerun reproduces the fit exactly
    and an interrupted epoch replays identically. Adam is used because the
    default ``lr=1e-3`` is an Adam learning rate; the optimiser and every
    hyper-parameter are echoed in the result so the report can state them
    instead of the reader having to trust a default.

    Returns top-1 and top-5 accuracy in percent, the final epoch's mean training
    loss, and the settings used.
    """
    device = torch.device(device)
    if hidden_dim < 1 or num_classes < 2:
        raise ValueError("hidden_dim must be positive and num_classes at least 2")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"invalid probe optimiser settings: lr={lr}, weight_decay={weight_decay}")
    train_labels = _labels(train_labels, num_classes, "train")
    val_labels = _labels(val_labels, num_classes, "validation")
    # Shuffled row reads dominate the cost at 1.28M rows, so the codes are
    # pinned on the GPU when they fit; batch composition is unaffected.
    train = CodeSource(train_codes, dim=hidden_dim, device=device, prefer_resident=True)
    validation = CodeSource(val_codes, dim=hidden_dim, device=device)
    if train.rows != len(train_labels):
        raise ValueError(f"{train.rows:,} train codes but {len(train_labels):,} labels")
    if validation.rows != len(val_labels):
        raise ValueError(f"{validation.rows:,} validation codes but {len(val_labels):,} labels")

    head = nn.Linear(hidden_dim, num_classes).to(device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    log = get_logger(progress_stage)
    targets = torch.as_tensor(train_labels, device=device)
    batches_per_epoch = math.ceil(train.rows / batch_size)
    progress = Progress(epochs * batches_per_epoch, stage=progress_stage, what="probe batches")
    final_loss = float("nan")
    started = time.perf_counter()
    # Probe gradients are float32 like the search; TF32 here would make the
    # probe's own numbers depend on the GPU generation rather than on the code.
    with fp32_matmul():
        head.train()
        for epoch in range(epochs):
            order = np.random.default_rng([seed, epoch]).permutation(train.rows)
            loss_sum = 0.0
            for start in range(0, train.rows, batch_size):
                # Sorting within the batch turns a scattered memory-map read
                # into a near-sequential one. The batch's mean loss does not
                # depend on row order, and the sort is deterministic, so this
                # costs nothing in reproducibility.
                rows = np.sort(order[start:start + batch_size])
                batch = train.dense_gather(rows)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(batch),
                                       targets[torch.as_tensor(rows, device=device)])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite probe loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(rows)
                progress.advance(1)
            final_loss = loss_sum / train.rows
            log.info("epoch %d/%d loss %.6f", epoch + 1, epochs, final_loss)
    train_seconds = time.perf_counter() - started

    head.eval()
    correct_1 = correct_5 = 0
    top5 = min(5, num_classes)
    with torch.no_grad(), fp32_matmul():
        for start in range(0, validation.rows, batch_size):
            stop = min(start + batch_size, validation.rows)
            logits = head(validation.dense(start, stop))
            truth = torch.as_tensor(val_labels[start:stop], device=device)
            ranked = logits.topk(top5, dim=1).indices
            correct_1 += int(ranked[:, 0].eq(truth).sum())
            correct_5 += int(ranked.eq(truth[:, None]).any(dim=1).sum())
    log.info("probe top1 %.2f%% top5 %.2f%% (%s)", 100.0 * correct_1 / validation.rows,
             100.0 * correct_5 / validation.rows, format_duration(train_seconds))
    return {"protocol": PROTOCOL,
            "top1": 100.0 * correct_1 / validation.rows,
            "top5": 100.0 * correct_5 / validation.rows,
            "correct_top1": correct_1, "correct_top5": correct_5,
            "queries": validation.rows, "train_rows": train.rows,
            "loss": final_loss, "epochs": epochs, "train_seconds": train_seconds,
            "settings": {"feature": "full_hidden_dim_latent", "feature_width": int(hidden_dim),
                         "num_classes": int(num_classes), "optimizer": "adam",
                         "lr": lr, "weight_decay": weight_decay, "schedule": "constant",
                         "batch_size": batch_size, "initialization": "zero",
                         # The codes arrive row-L2-normalised from
                         # encode_compact; the probe adds no further scaling.
                         "input_scaling": "row_l2_as_encoded",
                         "seed": seed, "precision": "float32_tf32_disabled",
                         "selection": "fixed final epoch; validation labels only score"}}


def _labels(labels: Any, num_classes: int, what: str) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{what} labels must be integer class ids, got {array.dtype}")
    if not len(array):
        raise ValueError(f"{what} labels are empty")
    if int(array.min()) < 0 or int(array.max()) >= num_classes:
        raise ValueError(f"{what} labels span [{array.min()}, {array.max()}], "
                         f"outside 0..{num_classes - 1}")
    return array.astype(np.int64, copy=False)
