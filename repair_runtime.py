"""Reproducible state and durable checkpoints for the staged repair experiment.

This module performs no work at import time. Checkpoints are local trusted
research artifacts: torch.load uses weights_only=False because RNG state and
optimizer metadata are part of the exact-resume contract.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import tempfile

from repair_data import atomic_json, canonical_hash, exclusive_lock, sha256_file


def named_parameters(sae, repair=None):
    result = {f"sae.{name}": value for name, value in sae.named_parameters()}
    if repair is not None:
        result.update({f"repair.{name}": value for name, value in repair.named_parameters()})
    if len({id(value) for value in result.values()}) != len(result):
        raise ValueError("Named optimizer parameters contain aliases")
    return result


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def pack_adam(optimizer, named):
    """Capture Adam moments by stable name, independent of parameter order."""
    if type(optimizer).__name__ != "Adam":
        raise ValueError("The controlled repair recipe requires torch.optim.Adam")
    names_by_id = {id(parameter): name for name, parameter in named.items()}
    if len(names_by_id) != len(named):
        raise ValueError("Duplicate named parameter identity")
    groups, seen = [], set()
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None or name in seen:
                raise ValueError("Optimizer contains an unnamed or duplicate parameter")
            names.append(name)
            seen.add(name)
        groups.append({"names": names, "options": _cpu_copy({key: value for key, value in group.items() if key != "params"})})
    if seen != set(named):
        raise ValueError("Some named parameters are missing from the optimizer")
    return {
        "schema": "named-adam-v1", "optimizer": "Adam", "groups": groups,
        "state": {name: _cpu_copy(optimizer.state.get(parameter, {})) for name, parameter in named.items()},
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in named.items()},
    }


def _same_option(first, second):
    # Optimizer constructor defaults are scalars, tuples, or None. Comparing
    # tensor learning rates is deliberately unsupported in this fixed recipe.
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(_same_option(a, b) for a, b in zip(first, second))
    return first == second


def restore_adam(optimizer, named, payload, *, allow_new=False):
    """Restore named moments, optionally adding zero-state repair parameters.

    Hyperparameters must already match the source optimizer. Existing moments
    are never mapped positionally, and allow_new only permits repair.* names.
    """
    import torch

    if payload.get("schema") != "named-adam-v1" or payload.get("optimizer") != "Adam" or type(optimizer).__name__ != "Adam":
        raise ValueError("Expected a named Adam checkpoint and Adam optimizer")
    old_state = payload["state"]
    missing = set(old_state) - set(named)
    added = set(named) - set(old_state)
    if missing or (added and not allow_new) or any(not name.startswith("repair.") for name in added):
        raise ValueError(f"Incompatible parameter names: missing={sorted(missing)}, added={sorted(added)}")
    old_options = {}
    for group in payload["groups"]:
        for name in group["names"]:
            if name in old_options:
                raise ValueError("Checkpoint repeats an optimizer parameter name")
            old_options[name] = group["options"]
    if set(old_options) != set(old_state):
        raise ValueError("Checkpoint parameter groups and state names disagree")
    names_by_id = {id(parameter): name for name, parameter in named.items()}
    current_options, seen = {}, set()
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != "params"}
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None or name in seen:
                raise ValueError("Optimizer parameter identity is missing or duplicated")
            seen.add(name)
            reference = old_options.get(name)
            if reference is None:
                # New correction parameters inherit the same recipe; there is
                # no independently tuned learning rate or weight decay.
                references = [entry["options"] for entry in payload["groups"]]
            else:
                references = [reference]
            if not any(set(options) == set(item) and all(_same_option(options[key], item[key]) for key in options) for item in references):
                raise ValueError(f"Adam hyperparameters differ for {name}")
            current_options[name] = options
    if seen != set(named):
        raise ValueError("Named parameters and optimizer membership disagree")
    restored = {}
    for name, parameter in named.items():
        if name in old_state:
            if list(parameter.shape) != payload["parameter_shapes"][name]:
                raise ValueError(f"Parameter shape changed for {name}")
            state = _cpu_copy(old_state[name])
        else:
            state = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format),
            }
            if current_options[name].get("amsgrad", False):
                state["max_exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step":
                device = parameter.device if current_options[name].get("capturable") or current_options[name].get("fused") else torch.device("cpu")
                state[key] = value.to(device=device)
            else:
                if value.shape != parameter.shape:
                    raise ValueError(f"Moment shape changed for {name}.{key}")
                state[key] = value.to(device=parameter.device, dtype=parameter.dtype)
        restored[parameter] = state
    # Validate every entry before mutating the optimizer.
    optimizer.state.clear()
    optimizer.state.update(restored)


def rng_state(mask_generator=None, *, extra_generators=None):
    """Capture global and explicitly independent random streams."""
    import numpy as np
    import torch

    generators = dict(extra_generators or {})
    if mask_generator is not None:
        if "mask" in generators:
            raise ValueError("The mask generator was supplied twice")
        generators["mask"] = mask_generator
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": {name: generator.get_state().cpu().clone() for name, generator in generators.items()},
        "generator_devices": {name: str(generator.device) for name, generator in generators.items()},
    }


def restore_rng(state, mask_generator=None, *, extra_generators=None):
    import numpy as np
    import torch

    generators = dict(extra_generators or {})
    if mask_generator is not None:
        if "mask" in generators:
            raise ValueError("The mask generator was supplied twice")
        generators["mask"] = mask_generator
    if set(generators) != set(state["generators"]):
        raise ValueError("Checkpoint and current independent RNG stream names differ")
    for name, generator in generators.items():
        if str(generator.device) != state["generator_devices"][name]:
            raise ValueError(f"RNG device changed for {name}; exact resume requires the same device")
    cuda_state = state.get("torch_cuda", [])
    if cuda_state and (not torch.cuda.is_available() or len(cuda_state) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG topology differs; exact resume is unavailable")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_state:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_state])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name].cpu())


def epoch_batches(ids, seed, epoch, batch_size, offset=0):
    """Yield (batch ordinal, row IDs) from a local (seed, epoch) permutation.

    offset is the next minibatch ordinal, not a count of rows. The last batch
    is retained. An unrelated mask/global RNG draw cannot alter this order.
    """
    import numpy as np

    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Expected one-dimensional integer row IDs")
    if batch_size < 1 or epoch < 0 or seed < 0 or offset < 0:
        raise ValueError("Batch size must be positive and seed/epoch/offset nonnegative")
    batches = (len(ids) + batch_size - 1) // batch_size
    if offset > batches:
        raise ValueError("Resume minibatch offset exceeds the epoch length")
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    order = generator.permutation(len(ids))
    for batch_number in range(offset, batches):
        start = batch_number * batch_size
        yield batch_number, ids[order[start:start + batch_size]]


def save_checkpoint(path, payload):
    """Atomically save a snapshot; interrupted writes never replace last.pt."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, *, map_location="cpu", expected_identity=None):
    """Load only a trusted locally produced checkpoint and verify its recipe."""
    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected a checkpoint mapping")
    if expected_identity is not None and canonical_hash(payload.get("identity")) != canonical_hash(expected_identity):
        raise ValueError("Checkpoint identity differs from this configuration/source/split/parent")
    return payload


def require_identity(path, identity):
    """Create or validate immutable JSON run identity before any fitting."""
    import json

    path = Path(path)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(current) != canonical_hash(identity):
            raise ValueError(f"Existing run identity differs: {path}")
    else:
        atomic_json(path, identity)


def source_hashes(paths):
    """Hash only explicit implementation files, not secrets or all workspace data."""
    result = {}
    for path in paths:
        path = Path(path)
        if path.name in result:
            raise ValueError("Source paths must have unique basenames")
        result[path.name] = sha256_file(path)
    return result
