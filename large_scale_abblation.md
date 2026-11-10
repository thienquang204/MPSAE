# Large-scale ablation: archived experiment settings

Snapshot taken before narrowing the main-result campaign. This file preserves the previous configuration, not measured results.

**Historical method definition (original 80-fit snapshot only):** at capture, `mpsaev2` meant the annealed
main-budget, OT-only variant B. The current implementation reuses that method
name for fixed K with OT **plus** instance contrastive loss. Copying the YAML
below into current code therefore restores the budget selection, but does not
reproduce the archived MPSAEv2 objective. Reproducing variant B requires its
historical implementation and compatible checkpoints. The YAML, resolved
presets and source hashes below remain the original snapshot.

Captured (UTC): 2026-11-10T04:20:52.149859+00:00

## Experiment selection

| Backbone | E1 training K / E2 inference K / downstream inference K | E2 training anchors | Downstream training K |
|---|---|---|---|
| ResNet-50 (`ff2048`) | 8, 64, 256, 1024 | 8, 64, 256 | 8 |
| ResNet-18 (`rn18`) | 8, 64, 256 | 8, 64, 256 | 8 |
| ConvNeXt (`convnext_tiny`) | 12, 48, 192 | 12, 48, 192 | 12 |
| ViT (`vit_b16`) | 12, 48, 192 | 12, 48, 192 | 12 |
| DINO (`dinov2_vits14`) | 12, 48, 192 | 12, 48, 192 | 12 |

Methods: `csr`, `csrv2`, `mpsae`, `mpsaev2`, `csr_multitopk`. Seed: `1227`.

80 source fits / E1 points; 75 E2 anchor curves with 240 points (including reused E1 points); 130 sparse + 8 dense downstream evaluations. EuroSAT uses all five backbones; SOP uses ResNet-50, ResNet-18 and ConvNeXt. Downstream evaluation is frozen, with no target-data training.

Existing precomputed embeddings are used; missing or incompatible bundles fail explicitly. E1 linear probing and retrieval timing are enabled. Timing uses 512 queries, 100 warm-up rounds and 2000 measured rounds.

## Exact configuration snapshots

The fenced YAML blocks below are copies of the active files before the main-result reduction. Comments and settings are preserved.

### `configs/thesis.yaml`

SHA256 of original file bytes: `98a0cd5939659bf7f7590803384ea1be9604e8ff45f70208534ba7c186df0b24`

```yaml
# Published embeddings only. Training presets remain in mpsae/config.py.
# HF coverage checked 2026-11-09: Flowers102 has no bundles; SOP has three.
name: thesis
backbones: [ff2048, rn18, convnext_tiny, vit_b16, dinov2_vits14]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18, convnext_tiny]
  train_k:                      # reuse one ImageNet checkpoint per method/backbone
    ff2048: [8]
    rn18: [8]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  mode: frozen
  revisions: {}                  # main is resolved once to an immutable commit
execution:
  storage_mode: direct_v1        # isolated work/direct-v1; no FFCV image copy
  precomputed_only: true        # missing/incompatible bundles fail without extracting images
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false            # metrics persist; avoid accumulating TB of codes
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### `configs/precomputed.yaml`

SHA256 of original file bytes: `a8ec05732ecbc049c8a0c7f112cc226a4afb87d31f198aac2396cbf51e2dd749`

```yaml
# Available in Lukita/MPSAE-embeddings on 2026-11-09: all five ImageNet and
# EuroSAT backbones, plus SOP for ff2048/rn18/convnext_tiny.
# Same fitting/evaluation and target coverage as thesis.yaml.
name: precomputed
backbones: [ff2048, rn18, convnext_tiny, vit_b16, dinov2_vits14]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18, convnext_tiny]
  train_k:
    ff2048: [8]
    rn18: [8]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:
    ff2048: [8, 64, 256, 1024]
    rn18: [8, 64, 256]
    convnext_tiny: [12, 48, 192]
    vit_b16: [12, 48, 192]
    dinov2_vits14: [12, 48, 192]
  mode: frozen
  revisions: {}
execution:
  storage_mode: direct_v1
  precomputed_only: true
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

## Resolved training presets and campaign plan

This snapshot includes the resolved backbone presets, method-specific batch sizes and optimizer/OT settings so later default changes do not silently redefine this archive. The two YAMLs differ only in campaign name/comments. Model and objective source code is not embedded; its configuration source hash is recorded below.

`mpsae/config.py` SHA256: `4036102f60b012cf62f1d991b80846932ab516cb21cad2ddd6fad52bdf2bd204`

```json
{
  "name": "thesis",
  "mode": "frozen",
  "plan_id": "fba8fc44dba1d82639a09c0ebd3de9fe4c6b3b6b7eee97d58a89a558539078e0",
  "source_cells": 80,
  "downstream_sparse_points": 130,
  "downstream_dense_points": 8,
  "config": {
    "name": "thesis",
    "backbones": [
      "ff2048",
      "rn18",
      "convnext_tiny",
      "vit_b16",
      "dinov2_vits14"
    ],
    "methods": [
      "csr",
      "csrv2",
      "mpsae",
      "mpsaev2",
      "csr_multitopk"
    ],
    "seeds": [
      1227
    ],
    "imagenet": {
      "enabled": true,
      "prepare": false,
      "train_k": {
        "ff2048": [
          8,
          64,
          256,
          1024
        ],
        "rn18": [
          8,
          64,
          256
        ],
        "convnext_tiny": [
          12,
          48,
          192
        ],
        "vit_b16": [
          12,
          48,
          192
        ],
        "dinov2_vits14": [
          12,
          48,
          192
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64,
          256,
          1024
        ],
        "rn18": [
          8,
          64,
          256
        ],
        "convnext_tiny": [
          12,
          48,
          192
        ],
        "vit_b16": [
          12,
          48,
          192
        ],
        "dinov2_vits14": [
          12,
          48,
          192
        ]
      },
      "e1": true,
      "e2": true,
      "linear_probe": true,
      "timing": true,
      "timing_query_batch": 512,
      "timing_warmup": 100,
      "timing_rounds": 2000,
      "timing_gallery_chunk": 32768,
      "timing_build_entry_cap": 4194304,
      "timing_retry_skipped": false
    },
    "downstream": {
      "enabled": true,
      "datasets": [
        "eurosat",
        "sop"
      ],
      "train_k": {
        "ff2048": [
          8
        ],
        "rn18": [
          8
        ],
        "convnext_tiny": [
          12
        ],
        "vit_b16": [
          12
        ],
        "dinov2_vits14": [
          12
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64,
          256,
          1024
        ],
        "rn18": [
          8,
          64,
          256
        ],
        "convnext_tiny": [
          12,
          48,
          192
        ],
        "vit_b16": [
          12,
          48,
          192
        ],
        "dinov2_vits14": [
          12,
          48,
          192
        ]
      },
      "mode": "frozen",
      "revisions": {},
      "backbones": {
        "sop": [
          "ff2048",
          "rn18",
          "convnext_tiny"
        ]
      }
    },
    "execution": {
      "train_missing": true,
      "continue_on_error": true,
      "retain_codes": false,
      "max_steps": 0,
      "storage_mode": "direct_v1",
      "cleanup_raw": false,
      "precomputed_only": true,
      "train_log_every": 10,
      "checkpoint_every": 100,
      "feature_residency": "auto"
    },
    "presets": {
      "ff2048": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "rn18": {
        "name": "paper-table9",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 1024,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 1.0,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "convnext_tiny": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "vit_b16": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "dinov2_vits14": {
        "name": "paper-table9",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 1024,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 1.0,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      }
    },
    "method_presets": {
      "ff2048": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "rn18": {
        "csr": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "convnext_tiny": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "vit_b16": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "dinov2_vits14": {
        "csr": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      }
    },
    "resolved_budgets": {
      "ff2048": {
        "imagenet_train_k": [
          8,
          64,
          256,
          1024
        ],
        "imagenet_eval_k": [
          8,
          64,
          256,
          1024
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8,
          64,
          256,
          1024
        ]
      },
      "rn18": {
        "imagenet_train_k": [
          8,
          64,
          256
        ],
        "imagenet_eval_k": [
          8,
          64,
          256
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8,
          64,
          256
        ]
      },
      "convnext_tiny": {
        "imagenet_train_k": [
          12,
          48,
          192
        ],
        "imagenet_eval_k": [
          12,
          48,
          192
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12,
          48,
          192
        ]
      },
      "vit_b16": {
        "imagenet_train_k": [
          12,
          48,
          192
        ],
        "imagenet_eval_k": [
          12,
          48,
          192
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12,
          48,
          192
        ]
      },
      "dinov2_vits14": {
        "imagenet_train_k": [
          12,
          48,
          192
        ],
        "imagenet_eval_k": [
          12,
          48,
          192
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12,
          48,
          192
        ]
      }
    }
  },
  "storage_estimate": {
    "imagenet_features_bytes": {
      "ff2048": 10915569400,
      "rn18": 2736879352,
      "convnext_tiny": 4099994360,
      "vit_b16": 4099994360,
      "dinov2_vits14": 2055321848
    },
    "total_imagenet_features_bytes": 23907759320,
    "active_writer_upper_bytes": 10915569400,
    "model_and_adam_tensor_bytes": 7621455360,
    "reserve_bytes": 10737418240,
    "excluded": "raw images, download/Arrow caches, model weights, environment, downstream features, sparse evaluation cache, checkpoint overhead"
  }
}
```

## Running this selection later

Copy the first YAML block to `configs/large_scale_abblation.yaml` and change its `name` to `large_scale_abblation` to give the reports a distinct campaign name. Review the resolved presets above if training defaults have changed. Inspect the plan before launching:

```bash
python -m mpsae --config configs/large_scale_abblation.yaml plan
```

Creating this archive does not schedule or launch an ablation. The active `thesis.yaml` and `precomputed.yaml` will select the smaller main-result experiment.


## Archived 35-fit main campaign before the three-backbone reduction

Captured (UTC): 2026-11-10T06:19:35.242227+00:00

This snapshot preserves the immediately preceding main experiment. It is distinct from the original 80-fit snapshot above. MPSAEv2 here is `fixed_ot_cl_v1`: fixed K with OT plus instance contrastive, without annealing.

| Backbone | E1 training / E2 inference / downstream inference K | Downstream training K |
|---|---|---|
| ResNet-50 | 8, 64 | 8 |
| ResNet-18 | 8, 64 | 8 |
| ConvNeXt | 12 | 12 |
| ViT | 12 | 12 |
| DINO | 12 | 12 |

Five methods, seed 1227: 35 source fits, 35 E2 anchor records / 55 points, 60 sparse + 8 dense downstream evaluations. EuroSAT uses all five backbones; SOP uses ResNet-50, ResNet-18 and ConvNeXt. Precomputed embeddings only.

### Previous `configs/thesis.yaml`

SHA256 of original file bytes: `2fadd48a623d7d7caf2945e4462517970e221e97aaec2f13ba2bd14bb6cb6517`

```yaml
# Main-result experiment. Previous grid: ../large_scale_abblation.md.
# Published embeddings only. Training presets remain in mpsae/config.py.
# HF coverage checked 2026-11-09: Flowers102 has no bundles; SOP has three.
name: thesis
backbones: [ff2048, rn18, convnext_tiny, vit_b16, dinov2_vits14]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18, convnext_tiny]
  train_k:                      # reuse one ImageNet checkpoint per method/backbone
    ff2048: [8]
    rn18: [8]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  mode: frozen
  revisions: {}                  # main is resolved once to an immutable commit
execution:
  storage_mode: direct_v1        # isolated work/direct-v1; no FFCV image copy
  precomputed_only: true        # missing/incompatible bundles fail without extracting images
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false            # metrics persist; avoid accumulating TB of codes
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### Previous `configs/precomputed.yaml`

SHA256 of original file bytes: `e850606adc348b23b60f0f44d911e563a1e99f64196fdfc5551b128e295f0e47`

```yaml
# Main-result experiment. Previous grid: ../large_scale_abblation.md.
# Available in Lukita/MPSAE-embeddings on 2026-11-09: all five ImageNet and
# EuroSAT backbones, plus SOP for ff2048/rn18/convnext_tiny.
# Same fitting/evaluation and target coverage as thesis.yaml.
name: precomputed
backbones: [ff2048, rn18, convnext_tiny, vit_b16, dinov2_vits14]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18, convnext_tiny]
  train_k:
    ff2048: [8]
    rn18: [8]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  eval_k:
    ff2048: [8, 64]
    rn18: [8, 64]
    convnext_tiny: [12]
    vit_b16: [12]
    dinov2_vits14: [12]
  mode: frozen
  revisions: {}
execution:
  storage_mode: direct_v1
  precomputed_only: true
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### Resolved 35-fit thesis plan and training presets

```json
{
  "name": "thesis",
  "mode": "frozen",
  "plan_id": "4f808915b60ca9cf4329d474de2d56248672b3027002ad1b648d29e83a91b2b3",
  "source_cells": 35,
  "downstream_sparse_points": 60,
  "downstream_dense_points": 8,
  "config": {
    "name": "thesis",
    "backbones": [
      "ff2048",
      "rn18",
      "convnext_tiny",
      "vit_b16",
      "dinov2_vits14"
    ],
    "methods": [
      "csr",
      "csrv2",
      "mpsae",
      "mpsaev2",
      "csr_multitopk"
    ],
    "seeds": [
      1227
    ],
    "imagenet": {
      "enabled": true,
      "prepare": false,
      "train_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8,
          64
        ],
        "convnext_tiny": [
          12
        ],
        "vit_b16": [
          12
        ],
        "dinov2_vits14": [
          12
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8,
          64
        ],
        "convnext_tiny": [
          12
        ],
        "vit_b16": [
          12
        ],
        "dinov2_vits14": [
          12
        ]
      },
      "e1": true,
      "e2": true,
      "linear_probe": true,
      "timing": true,
      "timing_query_batch": 512,
      "timing_warmup": 100,
      "timing_rounds": 2000,
      "timing_gallery_chunk": 32768,
      "timing_build_entry_cap": 4194304,
      "timing_retry_skipped": false
    },
    "downstream": {
      "enabled": true,
      "datasets": [
        "eurosat",
        "sop"
      ],
      "train_k": {
        "ff2048": [
          8
        ],
        "rn18": [
          8
        ],
        "convnext_tiny": [
          12
        ],
        "vit_b16": [
          12
        ],
        "dinov2_vits14": [
          12
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8,
          64
        ],
        "convnext_tiny": [
          12
        ],
        "vit_b16": [
          12
        ],
        "dinov2_vits14": [
          12
        ]
      },
      "mode": "frozen",
      "revisions": {},
      "backbones": {
        "sop": [
          "ff2048",
          "rn18",
          "convnext_tiny"
        ]
      }
    },
    "execution": {
      "train_missing": true,
      "continue_on_error": true,
      "retain_codes": false,
      "max_steps": 0,
      "storage_mode": "direct_v1",
      "cleanup_raw": false,
      "precomputed_only": true,
      "train_log_every": 10,
      "checkpoint_every": 100,
      "feature_residency": "auto"
    },
    "method_variants": {
      "mpsaev2": "fixed_ot_cl_v1"
    },
    "presets": {
      "ff2048": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "rn18": {
        "name": "paper-table9",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 1024,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 1.0,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "convnext_tiny": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "vit_b16": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "dinov2_vits14": {
        "name": "paper-table9",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 1024,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 1.0,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      }
    },
    "method_presets": {
      "ff2048": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "rn18": {
        "csr": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "convnext_tiny": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "vit_b16": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "dinov2_vits14": {
        "csr": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      }
    },
    "resolved_budgets": {
      "ff2048": {
        "imagenet_train_k": [
          8,
          64
        ],
        "imagenet_eval_k": [
          8,
          64
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8,
          64
        ]
      },
      "rn18": {
        "imagenet_train_k": [
          8,
          64
        ],
        "imagenet_eval_k": [
          8,
          64
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8,
          64
        ]
      },
      "convnext_tiny": {
        "imagenet_train_k": [
          12
        ],
        "imagenet_eval_k": [
          12
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12
        ]
      },
      "vit_b16": {
        "imagenet_train_k": [
          12
        ],
        "imagenet_eval_k": [
          12
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12
        ]
      },
      "dinov2_vits14": {
        "imagenet_train_k": [
          12
        ],
        "imagenet_eval_k": [
          12
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12
        ]
      }
    }
  },
  "storage_estimate": {
    "imagenet_features_bytes": {
      "ff2048": 10915569400,
      "rn18": 2736879352,
      "convnext_tiny": 4099994360,
      "vit_b16": 4099994360,
      "dinov2_vits14": 2055321848
    },
    "total_imagenet_features_bytes": 23907759320,
    "active_writer_upper_bytes": 10915569400,
    "model_and_adam_tensor_bytes": 3464163840,
    "reserve_bytes": 10737418240,
    "excluded": "raw images, download/Arrow caches, model weights, environment, downstream features, sparse evaluation cache, checkpoint overhead"
  }
}
```

Source hashes at capture (the working source includes the new MPSAEv2 objective):

- `mpsae/config.py`: `3c82c66d7cf8d6ea0564468f6498eb9f2e9f72036d5ee55e9c8665dcf7c6d466`
- `mpsae/methods.py`: `8ab1a441a83c86e2b940321f78f802476618a35e2338ef54a5267d870a7c0946`
- `mpsae/losses.py`: `b142b53862d9f793bd7f83ae8003142b87434de796052518535887bd8bffd27e`
- `mpsae/ot.py`: `6ee6357f49d5c207e4c1829c67b9753b424d4d2f279376fce3c84053271137ea`

To restore this 35-fit selection, copy the YAML from this section to a separate campaign file and give it a distinct name. The original 80-fit YAML above remains unchanged.


## Archived 20-fit campaign before ResNet-50 and ViT at K=8

Captured (UTC): 2026-11-10T08:54:19.977783+00:00

This is a settings snapshot, not measured results. It preserves ResNet-50
at K=8,64; ResNet-18 at K=8; and ViT at K=12. Five methods and seed
1227 give 20 source fits, 20 E2 anchor records / 30 points, and
35 sparse + 5 dense downstream evaluations. EuroSAT uses all three
backbones; SOP uses the two ResNets. MPSAEv2 uses the fixed-K OT plus
instance-contrastive objective for 15 epochs; other methods use 10.

### Previous `configs/thesis.yaml`

SHA256 of original file bytes: `ad71a354dcc1c28017804fe2c86a955b6a0ab107ffe75526f5fb9cac2720f027`

```yaml
# Main-paper first pass. Previous 35-fit setup: ../large_scale_abblation.md.
# Published embeddings only. Training presets remain in mpsae/config.py.
# Selected published bundles: ImageNet/EuroSAT for all three; SOP for ResNets.
name: thesis
backbones: [ff2048, rn18, vit_b16]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18]
  train_k:                      # reuse one ImageNet checkpoint per method/backbone
    ff2048: [8]
    rn18: [8]
    vit_b16: [12]
  eval_k:
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  mode: frozen
  revisions: {}                  # main is resolved once to an immutable commit
execution:
  storage_mode: direct_v1        # isolated work/direct-v1; no FFCV image copy
  precomputed_only: true        # missing/incompatible bundles fail without extracting images
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false            # metrics persist; avoid accumulating TB of codes
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### Previous `configs/precomputed.yaml`

SHA256 of original file bytes: `1e870f264dd99e6eb15606c744fba4b06da760599a02130de8778b298bd54326`

```yaml
# Main-paper first pass. Previous 35-fit setup: ../large_scale_abblation.md.
# Selected published embeddings: ImageNet/EuroSAT for ff2048/rn18/vit_b16;
# SOP for ff2048/rn18.
# Same fitting/evaluation and target coverage as thesis.yaml.
name: precomputed
backbones: [ff2048, rn18, vit_b16]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048, rn18]
  train_k:
    ff2048: [8]
    rn18: [8]
    vit_b16: [12]
  eval_k:
    ff2048: [8, 64]
    rn18: [8]
    vit_b16: [12]
  mode: frozen
  revisions: {}
execution:
  storage_mode: direct_v1
  precomputed_only: true
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### Resolved 20-fit thesis plan and training presets

```json
{
  "name": "thesis",
  "mode": "frozen",
  "plan_id": "2aa1d192ff437f2d2ef809542ca62734f8fa1fa7b4625df03424cc4a6c6a1173",
  "source_cells": 20,
  "downstream_sparse_points": 35,
  "downstream_dense_points": 5,
  "config": {
    "name": "thesis",
    "backbones": [
      "ff2048",
      "rn18",
      "vit_b16"
    ],
    "methods": [
      "csr",
      "csrv2",
      "mpsae",
      "mpsaev2",
      "csr_multitopk"
    ],
    "seeds": [
      1227
    ],
    "imagenet": {
      "enabled": true,
      "prepare": false,
      "train_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8
        ],
        "vit_b16": [
          12
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8
        ],
        "vit_b16": [
          12
        ]
      },
      "e1": true,
      "e2": true,
      "linear_probe": true,
      "timing": true,
      "timing_query_batch": 512,
      "timing_warmup": 100,
      "timing_rounds": 2000,
      "timing_gallery_chunk": 32768,
      "timing_build_entry_cap": 4194304,
      "timing_retry_skipped": false
    },
    "downstream": {
      "enabled": true,
      "datasets": [
        "eurosat",
        "sop"
      ],
      "train_k": {
        "ff2048": [
          8
        ],
        "rn18": [
          8
        ],
        "vit_b16": [
          12
        ]
      },
      "eval_k": {
        "ff2048": [
          8,
          64
        ],
        "rn18": [
          8
        ],
        "vit_b16": [
          12
        ]
      },
      "mode": "frozen",
      "revisions": {},
      "backbones": {
        "sop": [
          "ff2048",
          "rn18"
        ]
      }
    },
    "execution": {
      "train_missing": true,
      "continue_on_error": true,
      "retain_codes": false,
      "max_steps": 0,
      "storage_mode": "direct_v1",
      "cleanup_raw": false,
      "precomputed_only": true,
      "train_log_every": 10,
      "checkpoint_every": 100,
      "feature_residency": "auto"
    },
    "method_variants": {
      "mpsaev2": "fixed_ot_cl_v1"
    },
    "presets": {
      "ff2048": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "rn18": {
        "name": "paper-table9",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 1024,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 1.0,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      },
      "vit_b16": {
        "name": "paper-table3",
        "hidden_dim": 8192,
        "lr": 4e-05,
        "weight_decay": 0.0001,
        "adam_eps": 6.25e-10,
        "adam_betas": [
          0.9,
          0.999
        ],
        "epochs": 10,
        "batch_size": 4096,
        "k_aux": 512,
        "aux_coef": 0.03125,
        "nested_coef": 0.125,
        "cl_coef": 0.1,
        "cl_temperature": 0.2,
        "dead_threshold": 30,
        "k_decay_ratio": 0.7,
        "ot_weight": 2.0,
        "ot_mass": 0.9,
        "ot_eta": 0.2,
        "ot_microbatch": 32,
        "ot_max_iters": 400,
        "ot_max_refine_iters": 8000,
        "ot_tol": 0.0001
      }
    },
    "method_presets": {
      "ff2048": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 15,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "rn18": {
        "csr": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 15,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table9",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 1.0,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      },
      "vit_b16": {
        "csr": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csrv2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsae": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "mpsaev2": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 15,
          "batch_size": 1024,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        },
        "csr_multitopk": {
          "name": "paper-table3",
          "hidden_dim": 8192,
          "lr": 4e-05,
          "weight_decay": 0.0001,
          "adam_eps": 6.25e-10,
          "adam_betas": [
            0.9,
            0.999
          ],
          "epochs": 10,
          "batch_size": 4096,
          "k_aux": 512,
          "aux_coef": 0.03125,
          "nested_coef": 0.125,
          "cl_coef": 0.1,
          "cl_temperature": 0.2,
          "dead_threshold": 30,
          "k_decay_ratio": 0.7,
          "ot_weight": 2.0,
          "ot_mass": 0.9,
          "ot_eta": 0.2,
          "ot_microbatch": 32,
          "ot_max_iters": 400,
          "ot_max_refine_iters": 8000,
          "ot_tol": 0.0001
        }
      }
    },
    "resolved_budgets": {
      "ff2048": {
        "imagenet_train_k": [
          8,
          64
        ],
        "imagenet_eval_k": [
          8,
          64
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8,
          64
        ]
      },
      "rn18": {
        "imagenet_train_k": [
          8
        ],
        "imagenet_eval_k": [
          8
        ],
        "downstream_train_k": [
          8
        ],
        "downstream_eval_k": [
          8
        ]
      },
      "vit_b16": {
        "imagenet_train_k": [
          12
        ],
        "imagenet_eval_k": [
          12
        ],
        "downstream_train_k": [
          12
        ],
        "downstream_eval_k": [
          12
        ]
      }
    }
  },
  "storage_estimate": {
    "imagenet_features_bytes": {
      "ff2048": 10915569400,
      "rn18": 2736879352,
      "vit_b16": 4099994360
    },
    "total_imagenet_features_bytes": 17752443112,
    "active_writer_upper_bytes": 10915569400,
    "model_and_adam_tensor_bytes": 2644700160,
    "reserve_bytes": 10737418240,
    "excluded": "raw images, download/Arrow caches, model weights, environment, downstream features, sparse evaluation cache, checkpoint overhead"
  }
}
```

Source hashes at capture:

- `mpsae/config.py`: `aa150f9d86b6eb265200d375720f17c746a2f08bff96e546805a1f58ee54829d`
- `mpsae/methods.py`: `aced2459549c644bee8381de2965c8ebccc7f90521ef15fe254bbf6d73075e92`
- `mpsae/losses.py`: `b142b53862d9f793bd7f83ae8003142b87434de796052518535887bd8bffd27e`
- `mpsae/ot.py`: `6ee6357f49d5c207e4c1829c67b9753b424d4d2f279376fce3c84053271137ea`
- `mpsae/train.py`: `cf8c3ec505c107088c0b89299990a2d649e5549a534e825e0caa856649a73281`


## Archived 2026-11-10: K=8 source models with the eight-budget inference sweep

Before adding independent ResNet-50 fits at K=16/32/256, both active backbones
were trained at K=8 and evaluated at 2/4/8/16/32/64/128/256. This selection had
10 source fits, 80 ImageNet E2 points and 120 sparse + 3 dense downstream points.
Training recipes were unchanged: MPSAEv2 15 epochs, other methods 10 epochs.
The exact preceding YAML files follow; they are archival, not active defaults.

### configs/thesis.yaml

```yaml
# Two-setting main campaign. Previous 20-fit setup: ../large_scale_abblation.md.
# Published embeddings only. Training presets remain in mpsae/config.py.
# Selected published bundles: ImageNet/EuroSAT for both; SOP for ResNet-50.
name: thesis
backbones: [ff2048, vit_b16]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8]
    vit_b16: [8]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [2, 4, 8, 16, 32, 64, 128, 256]
    vit_b16: [2, 4, 8, 16, 32, 64, 128, 256]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048]
  train_k:                      # reuse one ImageNet checkpoint per method/backbone
    ff2048: [8]
    vit_b16: [8]
  eval_k:
    ff2048: [2, 4, 8, 16, 32, 64, 128, 256]
    vit_b16: [2, 4, 8, 16, 32, 64, 128, 256]
  mode: frozen
  revisions: {}                  # main is resolved once to an immutable commit
execution:
  storage_mode: direct_v1        # isolated work/direct-v1; no FFCV image copy
  precomputed_only: true        # missing/incompatible bundles fail without extracting images
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false            # metrics persist; avoid accumulating TB of codes
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```

### configs/precomputed.yaml

```yaml
# Two-setting main campaign. Previous 20-fit setup: ../large_scale_abblation.md.
# Selected published embeddings: ImageNet/EuroSAT for ff2048/vit_b16;
# SOP for ff2048.
# Same fitting/evaluation and target coverage as thesis.yaml.
name: precomputed
backbones: [ff2048, vit_b16]
methods: [csr, csrv2, mpsae, mpsaev2, csr_multitopk]
seeds: [1227]
imagenet:
  enabled: true
  prepare: false
  train_k:
    ff2048: [8]
    vit_b16: [8]
  eval_k:                       # E2 inference budgets; E1 evaluates at train_k
    ff2048: [2, 4, 8, 16, 32, 64, 128, 256]
    vit_b16: [2, 4, 8, 16, 32, 64, 128, 256]
  e1: true
  e2: true
  linear_probe: true
  timing: true
  timing_query_batch: 512
  timing_warmup: 100
  timing_rounds: 2000
  timing_gallery_chunk: 32768
  timing_build_entry_cap: 4194304
  timing_retry_skipped: false
downstream:
  enabled: true
  datasets: [eurosat, sop]
  backbones:
    sop: [ff2048]
  train_k:
    ff2048: [8]
    vit_b16: [8]
  eval_k:
    ff2048: [2, 4, 8, 16, 32, 64, 128, 256]
    vit_b16: [2, 4, 8, 16, 32, 64, 128, 256]
  mode: frozen
  revisions: {}
execution:
  storage_mode: direct_v1
  precomputed_only: true
  cleanup_raw: false
  train_missing: true
  continue_on_error: true
  retain_codes: false
  max_steps: 0
  train_log_every: 10
  checkpoint_every: 100
  feature_residency: auto
```
