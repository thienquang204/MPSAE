# Legacy ImageNet stage interface

For the native GPU installation and the combined ImageNet/downstream campaign,
use [server.md](server.md) and [benchmark.md](benchmark.md). This page describes
the older ImageNet-only stage/grid interface. Its direct Python examples assume
`MPSAE_CONFIG` is unset and omit `--config`. The shell launcher now defaults to
the YAML campaign; scientific subsets for that runner belong in YAML.

The operations manual behind [`README.md`](../README.md). The README says what
we claim and how to start a run; this file is what you read when a run stops,
when you want only part of the grid, or when you need to know whether
re-launching will redo four hours of work. Hardware and environment questions
belong to [`docs/server.md`](server.md) and are not repeated here.

## The mental model

Three invariants, and everything else follows from them.

**A stage runs once.** `mpsae.pipeline` has seven stages. Each is wrapped in
`logs.stage()`, which writes `work/state/<stage>.json` on the way out — success
or failure — and reads it on the way in. The receipt carries an `identity`: a
`canonical_hash` of a small dict describing what the stage was asked to do. A
stage is considered already done when its receipt has `ok: true` **and** its
stored `identity` equals the hash of the identity the current invocation would
use, and `--force` was not given. Anything else re-runs it.

For the seven top-level stages the identity includes `{version, options}`
(`pipeline.run_all`), with the resolved pipeline options. Changing the selection
changes the identity of every stage, so all seven receipts go stale at once.
Nothing is recomputed that was already materialised — the stages are
individually resumable below the receipt level — but you will see each stage
start again rather than print `skip (already done ...)`.

The `embed` stage additionally writes one receipt per backbone,
`work/state/embed-<backbone>.json`, whose identity is the thing that actually
determines the features: `{recipe, source, dim, normalize, input_size, seed,
smoke}`. Those are the receipts that make re-running `embed` cheap.

**A fit resumes.** `train.fit` saves `checkpoint.pt` every
`checkpoint_every = 100` optimiser steps, holding the model, the Adam state, the
CPU and CUDA RNG states and `next_step`. On entry it rebuilds a `signature` —
`canonical_hash(fit_identity(...))` over the method description, the preset, the
optimiser settings, the step schedule, the precision contract, the seed and the
embedding archive digests — and compares it with the signature stored in the
checkpoint. If they differ it **refuses**:

```
work/runs/.../checkpoint.pt belongs to a different recipe (a1b2... != c3d4...).
Move it aside rather than resuming across a settings change.
```

That is the intended behaviour, not a bug to work around. A checkpoint whose
learning rate, batch size, objective or source archive has changed is not a
partially trained version of the run you are asking for. `train.jsonl` is
truncated back to `next_step // log_every` records on resume, so a restart
cannot double-count metric rows.

**Evaluation is cached the same way.** `eval_e1.json` and `eval_e2.json` store
the fit `signature` they were computed from and are reused only when it still
matches. E2 is incremental: `eval_e2.json` is rewritten after every budget, so
an interrupted sweep resumes at the first missing K rather than the first K.
Encoded codes are memoised under `work/cache/codes/<hash>`, which is why the E2
sweep and the linear probe do not re-encode the gallery.

## The seven stages

In order. `pipeline.needed_stages` drops `ffcv` automatically when no selected
backbone uses the CSRv2 recipe.

| stage | does | reads | writes | time | fails when |
|---|---|---|---|---|---|
| `env` | preflight: GPU, compute capability vs CUDA runtime vs driver branch, imports, `HF_TOKEN`, free disk | `nvidia-smi`, the installed packages, the workspace | `state/env.json` (with the full environment fact sheet) | seconds | any problem at all — it collects them, logs each, then raises `N preflight problem(s)` |
| `imagenet` | materialises `ILSVRC/imagenet-1k` from the pinned parquet commit into `imagenet/{train,val}/n########/*.JPEG`, original JPEG bytes, never re-encoded | the gated HF dataset | `imagenet/`, `imagenet/_mpsae/prepare_<split>.json` | 2–5 h, network bound | token not accepted for the gated repo; a shard with no image bytes; row count ≠ 1,281,167 / 50,000; fewer than 1000 classes; validation not exactly 50/class |
| `ffcv` | serialises both split folders to `.ffcv` at `max_resolution=500`, `compress_probability=0.50`, `jpeg_quality=90` | `imagenet/{train,val}` | `ffcv/{train,val}_500_0.50_90.ffcv` + `.json` manifest | 1–2 h, CPU bound | `ffcv` not importable; `ImageFolder` and the folder scan disagree on image or class count; the existing file no longer matches its manifest's size and mtime |
| `embed` | extracts frozen embeddings per backbone (MRL recipe for `ff2048`, CSRv2 FFCV recipe for the rest), then measures dense 1-NN and gates on it | `imagenet/` or `ffcv/` | `embeddings/<backbone>/1K_{train,val}_<backbone>.npz` + `.npz.json`, `runs/<backbone>/dense_reference.json`, `state/embed-<backbone>.json` | 4–8 h for five backbones, JPEG-decode bound | no CUDA device (both recipes extract under fp16 autocast); a nonfinite or zero-norm feature row; dense 1-NN more than 0.3 pp from a published value |
| `publish` | uploads `embeddings/<backbone>/` to a private HF dataset repo | `embeddings/` | nothing locally | minutes to hours | only if the upload itself errors. **Skipped entirely when `HF_EMBEDDINGS_REPO` is unset**, which is the default — sending data off the box is never implied |
| `train` | the grid: fit every cell, evaluate E1, and at an E2 anchor sweep every budget | `embeddings/`, `cache/` | `runs/<backbone>/<method>/k<K>/seed<S>/*`, `runs/grid.json` | ~20–25 h of fitting, ~4 h E1, ~26 h E2 | first failing cell. It writes `failure.json` in that cell's directory and re-raises, so the stage stops rather than silently producing a hole in the grid |
| `report` | aggregates the run tree into tables and figures | `runs/` | `reports/tables/*`, `reports/figures/*` | minutes | essentially never: a partial grid is the normal state, every loader tolerates a missing file, and a figure with no data records why it was skipped instead of raising |

The phase times above are rough historical estimates, not A100 guarantees.

## Command line

`python -m mpsae [global options] <command>`. First source
`activate_server.sh` to use the installed environment. The shell
launcher is the campaign interface documented in `benchmark.md`; it saves a
console transcript, while credential checks happen only in stages that need
them. Metadata commands do not need a token.

### Commands

| command | effect |
|---|---|
| `run` | all seven legacy stages, minus `--skip`, minus the ones this configuration cannot use |
| `stage <name>` | exactly one stage, from `env imagenet ffcv embed publish train report`. Still receipt-gated |
| `status` | JSON: which receipts are green, how many grid cells have an `eval_e1.json`, and the shortfall per backbone and per method. No torch needed |
| `plan` | prints the grid and exits. No side effects, no workspace writes, no torch needed |
| `report` | rebuilds tables and figures from whatever is already in `runs/`, and prints the manifest. Takes `--out` to write somewhere other than `work/reports` |

### Options

Every flag below is from `mpsae/cli.py:build_parser`. The environment column is
the variable consulted for the default; `.env` is loaded into `os.environ`
first, and values already present in the environment win over the file.

| flag | env | default | change it when |
|---|---|---|---|
| `--env-file` | — | `.env` | the credentials live elsewhere. `pipeline.sh` passes `$MPSAE_ENV_FILE` |
| `--workspace` | `MPSAE_WORKSPACE` | `work` | the fast local disk is not under the repo — on Vast, `/workspace/mpsae` |
| `--device` | `MPSAE_DEVICE` | `cuda:0` | a second GPU, or two runs on one box |
| `--backbones` | `MPSAE_BACKBONES` | all five, branch 1 first | FFCV will not build (`ff2048` alone), or you want the paper-comparable branch finished first |
| `--methods` | `MPSAE_METHODS` | all five | isolating a comparison, e.g. `csr,mpsae` |
| `--seeds` | `MPSAE_SEEDS` | `1227,2026,2027` | a first pass at one seed. Note the `±` columns need ≥ 2 and the grid is specified at 3 |
| `--k` | `MPSAE_K` | each backbone's full `k_list` | a spot check, or finishing the E2 anchors first |
| `--workers` | `MPSAE_WORKERS` | `16` | the box has a different core count; this is the JPEG-decode worker count for both the FFCV writer and the extraction loaders |
| `--embed-batch-size` | `MPSAE_EMBED_BATCH` | `256` | under ~24 GB VRAM. `env` warns about this explicitly |
| `--hf-repo` | `HF_EMBEDDINGS_REPO` | unset | you want the archives published. Leaving it unset skips `publish` |
| `--public` | — | off (private) | essentially never: ImageNet-derived features inherit the dataset's non-commercial terms |
| `--wandb-project` | `WANDB_PROJECT` | unset | W&B also needs `WANDB_API_KEY`; without both, logging is off and `train.jsonl` is written regardless |
| `--smoke` | — | off | see below |
| `--max-steps` | — | `0` (full schedule); `20` under `--smoke` | capping a fit for a plumbing test. It is part of the fit identity, so a capped checkpoint will not resume into an uncapped run |
| `--no-e2` | — | E2 on | you only want the per-K comparison. E2 is most of the evaluation cost |
| `--no-probe` | — | probes on | the linear probe is the second most expensive thing per E1 point |
| `--force` | — | none | a stage's receipt is green but you want it redone. Takes stage names and `all` |
| `--skip` | — | none | leaving a stage out of a `run` |
| `--log-level` | `MPSAE_LOG_LEVEL` | `INFO` | `DEBUG` for step-level detail. The file handler always records DEBUG; this sets the console level |

### Credentials and other environment variables

| variable | read by | purpose |
|---|---|---|
| `HF_TOKEN` | `pipeline._env`, `embed.archive`, `data.imagenet` | required for gated ImageNet access, with accepted terms; write scope is needed only for the legacy publish stage |
| `HF_EMBEDDINGS_REPO` | `--hf-repo` default | setting it is what authorises the upload |
| `WANDB_API_KEY` | `logs.open_wandb` | W&B is enabled only when both this and a project are set, and a W&B failure never fails the pipeline |
| `WANDB_PROJECT` | `--wandb-project` default | as above |
| `HF_HOME` | `huggingface_hub` | point the Hub cache at the big disk, not `/` |
| `MPSAE_CUDA_HOME` | `setup_server.sh` | custom toolkit location matching the selected CUDA profile; preview with `--dry-run` |
| `MPSAE_ENV_FILE`, `MPSAE_PYTHON` | `pipeline.sh` | which `.env` and which interpreter |
| `PYTORCH_CUDA_ALLOC_CONF` | torch | optional allocator setting; set by the shell launcher, but not required by the installer |
| `NO_COLOR` | `logs.setup_logging` | suppress ANSI colour on the console handler |

Nothing from `.env` is ever written to a log or a manifest.

A few knobs are **not** reachable from the command line and have to be edited in
code: `RunSettings.gallery_chunk` / `query_block` / `encode_batch_size` /
`probe_epochs` / `dense_gate_tolerance` in `mpsae/runs.py`, and
`FitSettings.residency` / `checkpoint_every` / `log_every` in `mpsae/train.py`.
They are named in the troubleshooting table below because two of the error
messages tell you to lower them.

## The smoke path

```bash
python -m mpsae --smoke run
```

What it actually does:

* `imagenet` and `ffcv` return `{"skipped": "smoke"}` immediately. Nothing is
  downloaded and FFCV is never imported.
* `embed` calls `_synthesise_embeddings` instead of either real recipe: 50
  classes of class-conditional Gaussians, 40 rows per class for train (2,000
  rows) and 5 per class for validation (250 rows), at the backbone's true
  dimension, written through the same `finalize_archive` as a real archive.
* the dense 1-NN of those archives is measured and logged but **not** gated —
  `gate not applicable` — because there is no published number for Gaussians.
* `max_steps` defaults to `20`, so every fit is twenty optimiser steps.
* the `env` stage asks for 60 GiB of free disk instead of 500 GiB.
* `smoke` is part of both the stage identity and the fit identity, so a smoke
  run can never be mistaken for, or resumed into, a real one.
* `run_all` logs `smoke mode: tiny subsets and step caps. Results are not
  scientific.` and every synthetic manifest carries the deviation
  `SYNTHETIC DATA: smoke-test archive, not an ImageNet embedding`.

Its purpose is to exercise training, encoding, exact retrieval, the probe, the
timing harness, the aggregation and every figure end to end in about fifteen
minutes, on a box where nothing has run yet. Class-conditional Gaussians give a
dense 1-NN well above chance, so a wiring error shows up as a number near
chance rather than as a plausible result.

**A smoke number is not a measurement.** It is 2,000 synthetic rows and twenty
steps. Do not quote one, do not compare two of them, and do not leave smoke
runs in a `runs/` tree you intend to report from — `report` will aggregate them
without complaint, because it cannot tell from the run directory which archive
produced it.

## Running a subset

The grid is `backbone × K × method × seed`; any of the four can be narrowed, and
the narrowing composes.

```bash
# one backbone, and the only one that needs no FFCV
python -m mpsae --backbones ff2048 run

# the OT claim and its control against one baseline
python -m mpsae --methods csr,mpsae,csr_multitopk run

# the E2 anchors first, so the generalisation figure fills in early
python -m mpsae --backbones ff2048 --k 8,16,32,64,128,256 run

# one seed, no E2, no probes: the cheapest useful sweep
python -m mpsae --seeds 1227 --no-e2 --no-probe run

# a single cell, end to end
python -m mpsae --backbones rn18 --methods mpsaev2 --k 64 --seeds 2026 run
```

Two things to keep in mind. `--k` is intersected with each backbone's own
`k_list`, so `--k 8,64` is meaningful on `ff2048` and `rn18` and selects nothing
on the ×3 backbones, whose budgets are `3,6,12,...`. And E2 runs only at a
backbone's `e2_anchors`, so restricting `--k` to non-anchor budgets produces no
E2 points at all.

## `status` and `plan`

Both are torch-free by design — `command_plan` imports `config.grid` rather than
`mpsae.runs`, and `status` reads the result JSON directly rather than going
through the run machinery. You can plan and monitor from a laptop.

```
$ python -m mpsae plan
MPSAE 2.0.0: 690 training cells

backbone             d K budgets                                    E2 anchors              cells
-------------------------------------------------------------------------------------------------
ff2048            2048 2,4,8,16,32,64,128,256,512,1024,2048         8,16,32,64,128,256        165
rn18               512 2,4,8,16,32,64,128,256,512                   8,16,32,64,128,256        135
convnext_tiny      768 3,6,12,24,48,96,192,384,768                  3,6,12,24,48,96           135
vit_b16            768 3,6,12,24,48,96,192,384,768                  3,6,12,24,48,96           135
dinov2_vits14      384 3,6,12,24,48,96,192,384                      3,6,12,24,48,96           120
-------------------------------------------------------------------------------------------------
total                                                                                         690

methods: csr, csrv2, mpsae, mpsaev2, csr_multitopk
seeds:   1227, 2026, 2027
```

`status` prints the same shape of information about work actually done. A cell
counts as done when its `eval_e1.json` exists and contains a `top1`:

```
$ python -m mpsae status
{
  "stages": {"env": true, "imagenet": true, "ffcv": true, "embed": true,
             "publish": false, "train": false, "report": false},
  "cells": 690,
  "outstanding": 525,
  "done": 165,
  "by_backbone": {"ff2048": 0, "rn18": 135, "convnext_tiny": 135,
                  "vit_b16": 135, "dinov2_vits14": 120},
  "by_method": {"csr": 105, "csrv2": 105, "mpsae": 105,
                "mpsaev2": 105, "csr_multitopk": 105}
}
```

`by_backbone` and `by_method` count what is **outstanding**, not what is done.
A `false` under `stages` means "no green receipt", which includes "never run"
and "ran and failed" — the receipt itself distinguishes them.

## Resuming, forcing, redoing one thing

Re-running the same command is always the right first move. It is safe, and it
continues.

```bash
python -m mpsae run                  # continue legacy ImageNet work
python -m mpsae stage embed          # just the embeddings
python -m mpsae report               # just the tables and figures
```

To make something happen again:

```bash
python -m mpsae --force report run
python -m mpsae --force embed --backbones ff2048 run
python -m mpsae --force all run
python -m mpsae --skip publish,report run
```

`--force` is coarser than it looks on two stages, and the difference matters:

* `--force embed` invalidates the per-backbone `embed-<backbone>` receipts, so
  extraction runs again — but `FeatureSink` and `load_current_archive` still
  short-circuit on content, so an archive that already matches its manifest is
  not recomputed. To genuinely re-extract, move the archive and its `.json`
  aside.
* `--force train` sets `RunSettings.force`, which re-measures the dense
  reference and re-runs E1 and E2 for every cell. It does **not** retrain:
  `fit()` still returns early for any cell whose `summary.json` is complete and
  whose signature matches. To retrain a cell, delete (or move aside) its
  `checkpoint.pt` and `summary.json`.

Redoing one cell by hand is therefore:

```bash
mv work/runs/ff2048/mpsae/k64/seed1227 /tmp/
python -m mpsae --backbones ff2048 --methods mpsae --k 64 --seeds 1227 run
```

## Troubleshooting

Symptoms are the messages the code actually raises; grep for the quoted
fragment to find the site.

| symptom | cause | fix |
|---|---|---|
| `HF_TOKEN is unset; ILSVRC/imagenet-1k is gated` | `env` found no token | accept the terms at the dataset page with the account that owns the token, put it in `.env` |
| `cannot resolve dataset ILSVRC/imagenet-1k@main (...); accept the access conditions at ...` | the token exists but that account has not accepted the gate, or the token lacks read scope | same page, same account. There is no substitute dataset and the code will not silently pick one |
| `publishing needs a write-scoped HF_TOKEN` | `--hf-repo`/`HF_EMBEDDINGS_REPO` set, token not write-scoped | issue a write token, or unset the repo to skip `publish` |
| `ImageNet GPU extraction needs ffcv, cupy`, or `ffcv is not importable` | incomplete server environment or wrong interpreter | run `bash setup_server.sh`, then `source activate_server.sh`; see [server.md](server.md) |
| `sm_90 needs CUDA 12.1 or newer, but torch is built against 11.8` | wheel too old for the card | run the GPU-detecting `setup_server.sh` on that machine |
| `torch is a CUDA 13 build, which dropped sm_70` | a V100 needs a 12.x wheel | the native installer selects a CUDA 12.x profile |
| `driver branch 570 cannot load a CUDA 13.2 build (needs >= 580)` | driver below the runtime's floor | newer driver image from the provider, or a matching wheel. This would otherwise fail at the first kernel launch, minutes into a stage |
| `exact top-1 over 1,281,167 x 50,000 at dim=8192 needs ... plus 1.0 GiB reserve on cuda:0 (... free). Lower gallery_chunk, query_block, or the query split size.` | out of VRAM at search time | lower `RunSettings.gallery_chunk` (32768) and `query_block` (8192) in `mpsae/runs.py`. The check runs before the loop, so you lose nothing but the setup |
| `sparse retrieval score buffer needs ...` | the optional timing harness, not the metric | same two knobs, or set `RunSettings.measure_timing = False` |
| `codes stay memory-mapped (... needed, ... free)` in the log | informational. The probe wanted its codes resident and they did not fit | nothing. It changes speed, not values |
| `workspace has 180.0 GiB free, needs about 500.0 GiB` | `env` disk check | move `MPSAE_WORKSPACE` to the big disk. Once `embeddings/` exists, `imagenet/` and `ffcv/` are deletable — about 355 GB |
| `insufficient disk for <what> at <dir>: need N bytes plus M reserve, F free` | `io.require_disk`, raised by the FFCV writer, the feature sink, the code cache or an archive write | free space or move the workspace. Every such site checks before it starts writing |
| `ff2048: dense 1-NN 70.12% is -1.07 from the published 71.19%. The embedding recipe does not match the published one; fix it before training ...` | the extraction diverged from MRL's retrieval recipe | do not train through it. Every sparse number would then be measured against the wrong backbone and the comparison with the published tables would be meaningless. Check the weights, the resize/crop, the normalisation and the fp16 autocast contract in the archive manifest |
| `checkpoint.pt belongs to a different recipe (... != ...). Move it aside rather than resuming across a settings change.` | the fit identity changed — preset, optimiser, step count, seed, `--max-steps`, or the source archive digest | if the change was intended, move the cell directory aside and refit. If it was not, find out what changed: `config.json` in the cell holds the full identity that produced the checkpoint |
| `<cell>: checkpoint and summary disagree on the recipe` | `summary.json` and `checkpoint.pt` came from different attempts | delete both and refit the cell |
| `<archive>.npz hashes to ... but its manifest says ...; move both aside and extract again` | the archive changed under its manifest | re-extract. Nothing downstream will read a file whose digest does not match its receipt |
| `<ffcv file> changed since its manifest was written (N bytes vs M); delete both and write it again` | same, for the FFCV file | delete the `.ffcv` and its `.json` |
| `cursor 12345 is not a whole number of batches; the FFCV pass can only be resumed on a batch boundary` | an interrupted CSRv2 extraction with a partial batch in the sink | delete that backbone's `cache/features/<signature>/` and re-extract the split |
| `<receipt> was written for a different ImageNet release; remove the folder and the receipt to start over` | the pinned dataset commit moved between attempts | start that split over, or restore the previous revision |
| `nonfinite feature rows at [...]`, `zero-norm feature rows at [...]` | a bad extraction batch | fatal on purpose: every metric L2-normalises, which is undefined at zero, and a NaN would reach the SAE weights without a trace. Re-extract |
| `nonfinite csr objective: {...}` (`FloatingPointError`) | the fit diverged | check `train.jsonl` for the step before. The learning rate is the papers' 4e-5, not the release's 4e-4; a divergence here is a real finding, not a knob |
| `partial OT infeasible after bounded refinement: {...}` | the Greenkhorn solve did not reach the mass/cap tolerance even after refinement | the `ot_*` fields in `train.jsonl` carry `mass_error`, `cap_violation` and `iterations`; raise `ot_max_refine_iters` or `ot_tol` in the preset only with a reason written down |
| `missing train embeddings for ff2048: <path>. Run the embedding stage ...` | `train` or `report` reached a backbone with no archive | run `stage embed` for it, or download the published archive |
| `epoch ended before its batches were exhausted` | the batch source returned fewer batches than the schedule expects | a bug, not a configuration problem. Capture the cell and the step count |
| a stage prints `skip (already done ...)` when you wanted it to run | the receipt's identity matches | `--force <stage>`, and read the note above about what `--force` does and does not reach |
| `N run(s) outside the configured grid` in the report log | `runs/` holds directories the current grid does not contain — usually a leftover from a narrower or an older configuration | harmless for the tables, but check it is not a stale smoke run |

## Long runs

```bash
tmux new -s mpsae
source activate_server.sh
python -m mpsae run
# C-b d to detach, tmux attach -t mpsae to come back
```

Four kinds of record, and they answer different questions.

* `work/logs/console-<timestamp>.log` — the raw transcript of one invocation,
  teed by `pipeline.sh`. This is what survives a crash, an OOM-kill or a
  disconnect, and the only place that captures output from anything that is not
  our logger (pip, FFCV, tracebacks from a hard failure).
* `work/logs/<command>.log` and `work/logs/pipeline.log` — the structured
  logger, at DEBUG regardless of `--log-level`, with the stage name in every
  line. A `run` writes both `run.log` and `pipeline.log`; `stage embed` writes
  `stage.log` and `embed.log`.
* `work/state/*.json` — what finished, when, how long it took, and on failure
  the exception that stopped it. This is the first file to read after an
  interruption.
* `work/runs/**/train.jsonl` — one record per logged step, written whether or
  not W&B is configured. The field list is in
  [`docs/outputs.md`](outputs.md#trainjsonl).

Progress lines come from `logs.Progress`, which reports at most every 30 s with
a rate and an ETA, and always on the final item. A long silence in the console
with the GPU busy is normal during a gallery search; a long silence with the GPU
idle is not.

`status` is cheap, needs no GPU and can be run from another shell at any time.
`report` can be run mid-grid: every table and figure renders from whatever
exists, carries its own `n`, and names the seeds it is missing.
