# Large-scale ablation: archived experiment settings

Snapshot taken before narrowing the main-result campaign. This file preserves the previous configuration, not measured results.

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
