"""Supervised linear probes on frozen representations, independently at each K.

This is classification, not the retrieval top-1 metric. Only training labels
fit the probes. Endpoint query labels score them once; there is no early
stopping or selection on those labels. Sparse codes stay compact on disk and
only a minibatch is expanded to SAE width. The encoder is never optimized.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from repair_data import atomic_json, canonical_hash, sha256_file
from repair_runtime import epoch_batches, load_checkpoint, save_checkpoint


PROBE_PROTOCOL = "frozen-linear-probe-v1"


def code_batch(codes, positions, width, device):
    """Expand one compact batch; retain original dictionary coordinate IDs."""
    values = torch.as_tensor(np.array(codes["values"][positions], dtype=np.float32, copy=True), device=device)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite probe features")
    if "indices" not in codes:
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("Dense probe feature width changed")
        return values
    indices = torch.as_tensor(np.array(codes["indices"][positions], dtype=np.int64, copy=True), device=device)
    if indices.shape != values.shape or bool((indices < 0).any()) or bool((indices >= width).any()):
        raise ValueError("Invalid compact probe coordinates")
    return values.new_zeros((len(values), width)).scatter_(1, indices, values)


def fit_linear_probe(train_codes, train_labels, query_codes, query_labels, *, width,
                     num_classes, device, epochs=10, batch_size=1024, lr=.1,
                     weight_decay=1e-4, seed=1227, checkpoint_path=None, identity=None):
    """Fit one independent softmax classifier; supports CPU contract tests.

    Epoch checkpoints include the SGD momentum and cumulative training time.
    An interrupted partial epoch is replayed from its last committed boundary.
    Zero initialization avoids perturbing global RNG and is shared across K.
    """
    device = torch.device(device)
    if width < 1 or num_classes < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid probe dimensions or schedule")
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Invalid probe optimizer settings")
    train_labels = np.asarray(train_labels)
    query_labels = np.asarray(query_labels)
    if train_labels.dtype.kind not in "iu" or query_labels.dtype.kind not in "iu":
        raise ValueError("Probe labels must be integer class IDs")
    for codes, labels in ((train_codes, train_labels), (query_codes, query_labels)):
        if labels.shape != (len(codes["values"]),) or not len(labels):
            raise ValueError("Probe labels must align with nonempty feature rows")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError("Probe labels outside the declared class range")
    train_labels = train_labels.astype(np.int64, copy=False)
    query_labels = query_labels.astype(np.int64, copy=False)
    # nn.Linear's default initializer consumes RNG; create it in a private fork.
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(width, num_classes)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    head.to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=.9, weight_decay=weight_decay)
    first_epoch, training_seconds, final_loss = 0, 0., None
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path, expected_identity=identity)
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch = int(saved["completed_epochs"])
        training_seconds = float(saved["train_seconds"])
        final_loss = saved["loss"]
        if not 0 <= first_epoch <= epochs or not math.isfinite(training_seconds) or training_seconds < 0:
            raise ValueError("Invalid linear-probe checkpoint cursor or timer")
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    positions = np.arange(len(train_labels), dtype=np.int64)
    for epoch in range(first_epoch, epochs):
        rate = lr * .5 * (1. + math.cos(math.pi * epoch / epochs))
        for group in optimizer.param_groups:
            group["lr"] = rate
        synchronize()
        started = time.perf_counter()
        loss_sum = 0.
        head.train()
        for _, rows in epoch_batches(positions, seed, epoch, batch_size):
            batch = code_batch(train_codes, rows, width, device)
            targets = torch.as_tensor(train_labels[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite linear-probe loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
        synchronize()
        training_seconds += time.perf_counter() - started
        final_loss = loss_sum / len(train_labels)
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, {"identity": identity, "head": head.state_dict(),
                "optimizer": optimizer.state_dict(), "completed_epochs": epoch + 1,
                "train_seconds": training_seconds, "loss": final_loss})
        print(f"Linear probe width={width} epoch={epoch+1}/{epochs} loss={final_loss:.6f}", flush=True)
    correct = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, len(query_labels), batch_size):
            rows = slice(start, start + batch_size)
            logits = head(code_batch(query_codes, rows, width, device))
            targets = torch.as_tensor(query_labels[rows], device=device)
            correct += int(logits.argmax(dim=1).eq(targets).sum())
    return {"top1": 100. * correct / len(query_labels), "correct": correct,
            "total": len(query_labels), "train_examples": len(train_labels),
            "train_seconds": training_seconds, "loss": final_loss, "epochs": epochs,
            "feature_width": width, "num_classes": num_classes,
            "train_time_scope": "classifier updates and batch transfers; excludes representation encoding, query scoring and checkpoint I/O",
            "empty_code_policy": "classifier bias prediction; no special retrieval-miss rule"}


def evaluate_linear_probes(args, bundle, model, output, checkpoint_identity):
    """Endpoint per-K classifiers plus full-parent classifier, without FAISS.

    Existing retrieval encodings are reused through the fingerprint-validated
    encoding cache. The parent classifier is included even when dense-parent
    retrieval is disabled; it streams features without a resident dense index.
    """
    from mmrsae_imagenet import evaluation_budgets, retrieval_config
    from repair_benchmark import _encoded_cache

    device = next(model.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Production linear probes require the selected CUDA device")
    output = Path(output)
    directory = output / "classification" / "endpoint"
    directory.mkdir(parents=True, exist_ok=True)
    fit_ids = np.asarray(bundle.fit_ids, dtype=np.int64)
    query_ids = np.asarray(bundle.query_ids, dtype=np.int64)
    if bundle.query is None and np.intersect1d(fit_ids, query_ids).size:
        raise ValueError("Linear probe fitting and query IDs must be disjoint")
    if not np.array_equal(fit_ids, bundle.gallery_ids):
        raise ValueError("Probe train rows must match the controlled retrieval gallery")
    query_features = bundle.train if bundle.query is None else bundle.query
    query_labels = bundle.train_labels if bundle.query_labels is None else bundle.query_labels
    train_labels = np.asarray(bundle.train_labels)[fit_ids]
    targets = np.asarray(query_labels)[query_ids]
    classes = int(np.max(np.asarray(bundle.train_labels))) + 1
    budgets = evaluation_budgets(args, model.config.input_dim)
    identity = {"protocol": PROBE_PROTOCOL, "checkpoint_sha256": checkpoint_identity,
        "data": canonical_hash(bundle.provenance["identity"]), "evaluation_topk": budgets,
        "fit_rows": canonical_hash(fit_ids.tolist()), "query_rows": canonical_hash(query_ids.tolist()),
        "epochs": args.probe_epochs, "batch_size": args.probe_batch_size,
        "lr": args.probe_lr, "weight_decay": args.probe_weight_decay, "momentum": .9,
        "seed": args.seed, "normalization": "row_l2", "initialization": "zero",
        "schedule": "epoch cosine", "selection": "fixed final epoch; query labels used only for scoring",
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "code": sha256_file(Path(__file__))}
    path = directory / "result.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("identity") != identity:
            raise ValueError("Existing linear-probe results belong to a different experiment")
        if result.get("complete"):
            return result
    else:
        result = {"protocol": PROBE_PROTOCOL, "identity": identity, "per_topk": {},
                  "full_parent": None, "complete": False,
                  "interpretation": "supervised linear probes on frozen codes; distinct from 1-NN retrieval"}
    prior_mode = model.training
    model.eval()
    try:
        for k in [*budgets, None]:
            key = str(k) if k is not None else "full_parent"
            already_complete = key in result["per_topk"] if k is not None else result["full_parent"] is not None
            if already_complete:
                continue
            dense = k is None
            metric_path = output / ("parent/endpoint/result.json" if dense else f"retrieval/endpoint/k_{k}/result.json")
            retrieval = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None
            encoding_identity = (retrieval["identity"] if retrieval else
                                 {"probe": identity, "method": "full_parent" if dense else args.variant})
            if retrieval and not dense and encoding_identity.get("checkpoint_sha256") != checkpoint_identity:
                raise ValueError("Linear probe cannot reuse a different encoder checkpoint")
            config = {"model": {"k": args.k if dense else k}, "retrieval": retrieval_config(args)}
            cache_root = (output / ("parent/encoded_cache" if dense else "retrieval/endpoint/encoded_cache"))
            query_role = "training_queries" if bundle.query is None else "validation_queries"
            with torch.no_grad():
                train, train_meta = _encoded_cache(bundle.train, fit_ids, None if dense else model.sae,
                    None, config, cache_root, encoding_identity, "training_gallery", device)
                queries, query_meta = _encoded_cache(query_features, query_ids, None if dense else model.sae,
                    None, config, cache_root, encoding_identity, query_role, device)
            print(f"Linear probe {args.variant} K={key}", flush=True)
            point = fit_linear_probe(train, train_labels, queries, targets,
                width=model.config.input_dim if dense else model.config.hidden_dim,
                num_classes=classes, device=device, epochs=args.probe_epochs,
                batch_size=args.probe_batch_size, lr=args.probe_lr,
                weight_decay=args.probe_weight_decay, seed=args.seed,
                checkpoint_path=directory / key / "last.pt", identity={**identity, "K": k})
            point.update(K=k, device_name=identity["device_name"],
                actual_nnz_mean=None if dense else train_meta["nonzero_sum"] / len(fit_ids),
                query_actual_nnz_mean=None if dense else query_meta["nonzero_sum"] / len(query_ids),
                actual_nnz_scope="training_gallery", representation="full_parent" if dense else args.variant)
            if dense:
                result["full_parent"] = point
            else:
                result["per_topk"][key] = point
            atomic_json(path, result)
        result["complete"] = True
        atomic_json(path, result)
        return result
    finally:
        model.train(prior_mode)
