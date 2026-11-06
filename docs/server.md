# Server notes

Written for a rented box (Vast.ai and similar). `scripts/setup_server.sh`
automates all of this; the reasoning is here so a failure is diagnosable.

## Minimum configuration

| | requirement | why |
|---|---|---|
| arch | x86_64 | FFCV and the torch wheels we use |
| OS | Ubuntu ≥ 22.04 | libjpeg-turbo / opencv packaging |
| CUDA | ≥ 12.1, **≥ 12.8 on Blackwell** | see below |
| GPU | ≥ 24 GB VRAM, 40–80 GB comfortable | the FF2048 fitting matrix is 10.5 GB and we keep it resident |
| RAM | ≥ 64 GB, 128 GB comfortable | JPEG decode workers plus a host-resident fallback |
| CPU | ≥ 16 cores | embedding extraction is JPEG-decode bound, not GPU bound |
| disk | ≥ 500 GB SSD | ImageNet ≈ 155 GB + folders + FFCV + embeddings + 690 checkpoints |
| network | stable | the ImageNet download, and the optional embedding upload |

## CUDA and architectures

Two independent constraints, and `setup_server.sh` resolves both:

* **the driver sets the ceiling.** CUDA 13.x needs driver branch ≥ 580, CUDA
  12.8 needs ≥ 570, older 12.x needs ≥ 525. The wheel ships its own runtime, so
  the host toolkit is irrelevant — only the driver matters.
* **the GPU sets the floor.** CUDA 13 dropped Maxwell, Pascal and Volta
  outright: it compiles only for sm_75 and above. A V100 therefore needs a 12.x
  build no matter how new the driver is.

Wheel indexes, checked 2026-11:

| index | newest torch | note |
|---|---|---|
| `cu126` | 2.14.1 | the fallback for pre-Turing GPUs and old drivers |
| `cu128` | **2.11.0 — frozen** | only worth using for Blackwell on driver 570–579 |
| `cu129` | 2.13.0 | |
| `cu130` | 2.14.1 | first CUDA 13 line |
| `cu132` | **2.15.0** | current; the default once the driver is ≥ 580 |

| GPU | capability | picked |
|---|---|---|
| V100 | sm_70 | `cu126` (CUDA 13 dropped it) |
| A100 | sm_80 | `cu132`, else `cu126` |
| L40S, 4090 | sm_89 | `cu132`, else `cu126` |
| H100 | sm_90 | `cu132`, else `cu126` |
| B200 | sm_100 | `cu132`, else `cu128` |
| RTX 5090, RTX PRO 6000 | sm_120 | `cu132`, else `cu128` |

**So CUDA 13 is the good case**, not a problem: with driver ≥ 580 you get
`cu132` and the current torch. The awkward case is the opposite one — Blackwell
on a driver stuck at 57x, which can only load `cu128`, and that index has not
moved past torch 2.11.0. If the provider offers a newer driver image, take it.

Override the choice when you need to:

```bash
MPSAE_TORCH_INDEX=https://download.pytorch.org/whl/cu130 ./scripts/setup_server.sh
```

`TORCH_CUDA_ARCH_LIST` is generated from the capabilities actually present
rather than hard-coded, because CUDA 13's `nvcc` *rejects* a dropped
architecture instead of ignoring it. It only affects the optional `csrc/`
extension.

A mismatched wheel does not fail at import — it fails at the first kernel
launch, often minutes into a stage. `stage_env` checks the GPU, the runtime and
the driver branch together before anything downloads, and `setup_server.sh`
runs a real matmul as proof.

## FFCV

Needed only by the CSRv2 embedding recipe, i.e. `rn18`, `convnext_tiny`,
`vit_b16`, `dinov2_vits14`. The `ff2048` backbone uses MRL's torchvision
recipe and does not need it.

It has no wheels and compiles against libjpeg-turbo and opencv. If pip fails:

```bash
conda create -n mpsae python=3.11 -y && conda activate mpsae
conda install -c conda-forge -c pytorch cupy pkg-config libjpeg-turbo opencv numba -y
pip install ffcv
```

Or proceed without it:

```bash
./scripts/pipeline.sh -- --backbones ff2048
```

This is the single most likely setup failure, which is why `stage_env` imports
`ffcv` before anything downloads and only when a backbone actually needs it.

## Disk layout

Everything lives under one workspace (`MPSAE_WORKSPACE`, default `./work`):

```
work/
  imagenet/    train/n########/*.JPEG, val/...   ~155 GB
  ffcv/        train_500_0.50_90.ffcv, val_...   ~200 GB
  embeddings/  <backbone>/1K_{train,val}_*.npz   ~22 GB total
  cache/       feature mmaps, encoded-code cache
  runs/        690 × {checkpoint.pt, train.jsonl, eval_*.json}
  reports/     CSV, Markdown, PNG, PDF, captions
  logs/        pipeline.log, console-*.log
  state/       stage receipts (what lets a restart skip work)
```

Put the workspace on the instance's fast local disk, not on a network volume:
training reads the fitting matrix repeatedly. On Vast that usually means
`/workspace`, so set `MPSAE_WORKSPACE=/workspace/mpsae` and
`HF_HOME=/workspace/hf` in `.env` to keep the Hugging Face cache off the small
root filesystem.

Once `embeddings/` exists (and ideally has been published), `imagenet/` and
`ffcv/` can be deleted — about 355 GB back. Nothing downstream reads them.

## Expected cost

Rough, for one H100, assuming the fitting matrix stays resident on the GPU:

| phase | time |
|---|---|
| ImageNet download + materialise | 2–5 h, network bound |
| FFCV serialisation | 1–2 h, CPU bound |
| embedding extraction, 5 backbones × 1.33 M images | 4–8 h, JPEG-decode bound |
| training, 690 fits | ~20–25 h |
| E1 evaluation, 690 points | ~4 h |
| E2 evaluation, ~4100 points | ~26 h |

Search dominates evaluation because it is exact at the dictionary width
(h = 8192): 50 k queries × 1.28 M gallery × 8192 ≈ 1 PFLOP per point, about
20 s in FP32. Reducing E2's seeds or anchors is the lever if that is too much.

## Running long jobs

```bash
tmux new -s mpsae
./scripts/pipeline.sh
# C-b d to detach; tmux attach -t mpsae to return
```

`work/logs/console-*.log` keeps the transcript of every invocation, and
`work/runs/**/train.jsonl` holds the structured metrics regardless of whether
W&B is configured. `./scripts/pipeline.sh status` is cheap and needs no GPU.
