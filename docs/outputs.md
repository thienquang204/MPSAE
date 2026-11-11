# Artefacts

The schema document. Everything the pipeline writes, field by field, so that a
number found in `work/` six months from now can be traced back to how it was
produced. The operational side — which stage writes what, when, and how to
re-run it — is in [`docs/pipeline.md`](pipeline.md); the claims the numbers are
for are in [`README.md`](../README.md).

## The workspace

One root, `MPSAE_WORKSPACE`, default `./work`. Created by `pipeline.Paths`.

```
work/
  imagenet/     train/n########/*.JPEG and val/..., original JPEG bytes          ~155 GB
  imagenet/_mpsae/  our scratch: pinned-release receipts, per-split progress
  ffcv/         train_500_0.50_90.ffcv, val_... plus one .json manifest each     ~200 GB
  embeddings/   <backbone>/1K_{train,val}_<backbone>.npz and .npz.json            ~22 GB
  cache/        feature mmaps, extraction sinks, encoded-code cache, hub blobs
  runs/         one directory per grid cell, plus dense_reference.json and grid.json
  reports/      tables/ and figures/
  logs/         <command>.log, pipeline.log, console-<timestamp>.log
  state/        one receipt per stage, plus options.json
```

Once `embeddings/` exists, `imagenet/` and `ffcv/` are no longer read by
anything and can be deleted.

Every JSON file in the tree is written through `io.atomic_json`: temporary file,
fsync, rename. A half-written artefact is therefore not a state the pipeline can
leave behind, and a file that exists is complete.

## `runs/<backbone>/<method>/k<K>/seed<S>/`

The directory segment matches the active logical method name: `csr`, `csrv2`
or `mpsae`. Existing CSR multi-TopK and MPSAEv2 directories remain historical
artefacts; they are excluded from experiment selection and new reports.
Their old identities and training-duration rules are documented in
[the historical note](notes/excluded_methods.md).

Six possible files. `config.json`, `train.jsonl`, `checkpoint.pt` and
`summary.json` come from the fit; `eval_e1.json` from the per-K measurement;
`eval_e2.json` only at an E2 anchor budget; `failure.json` only if the cell
raised.

### `config.json`

Written by `train.fit` before the first step, so it exists even for a fit that
never finished. It is the authoritative record of what was asked for.

| field | meaning |
|---|---|
| `signature` | `canonical_hash(identity)`. The value every later artefact is matched against |
| `identity` | the full recipe, expanded below |
| `cell` | `{backbone, method, k, seed}` |
| `code_version`, `backbone`, `method`, `k`, `seed` | the same facts, flat, for cheap reads |
| `total_steps` | `batches_per_epoch × epochs`, capped by `--max-steps` |
| `log_every` | metric stride, `10` |

`identity` (also stored inside `checkpoint.pt` and `summary.json`):

| field | meaning |
|---|---|
| `format` | `mpsae-fit-v2`, the checkpoint layout |
| `code_version` | `mpsae.__version__` at fit time |
| `backbone`, `method`, `target_k`, `seed` | the cell |
| `objective` | `methods.describe(...)`: loss composition, gate budgets, regularisers and their weights, and OT settings where applicable (`weight`, `cost`, `mass`, `eta`, `microbatch`, `max_iters`, `tol`, `views`) |
| `optimizer` | `{name: Adam, lr, betas, eps, weight_decay, schedule: constant, grad_clip: null, decay_applies_to}`. The last field records that decay is coupled L2 over *all* parameters including biases, as the release does — not AdamW's decoupled form |
| `schedule` | `{epochs, batch_size, batches_per_epoch, total_steps, drop_last: false}` |
| `precision` | `{params: float32, autocast: false, tf32: false}` |
| `data` | `bundle.identity`: per split, the archive `sha256`, `rows`, `recipe` and `source` |

Because `data` carries the archive digests, re-extracting an embedding changes
every fit signature that reads it. That is the point: a checkpoint is tied to
the exact features it saw.

### `summary.json`

Written once, after the last step. Its presence with `complete: true` and a
matching `signature` is what makes `fit()` return early.

| field | meaning |
|---|---|
| `identity`, `signature` | as above |
| `complete` | always `true` in a written summary; checked explicitly rather than inferred from the file existing |
| `cell` | `{backbone, method, k, seed}` |
| `steps` | optimiser steps actually taken |
| `train_seconds` | **the authoritative wall-clock total for the fit**, summed across resumptions |
| `gpu_hours` | `train_seconds / 3600` |
| `steps_per_second` | `steps / train_seconds` |
| `peak_memory_bytes` | `torch.cuda.max_memory_allocated` for this fit, or `null` off CUDA |
| `residency` | `cuda`, `cpu` or `mmap` — where the fitting matrix lived. A performance fact, not a numerical one |
| `final_metrics` | the last step's metric dict, same keys as a `train.jsonl` record minus `epoch`/`batch`/`lr`/`rows`/`elapsed` |
| `parameters` | total parameter count |

### `eval_e1.json`

One model trained at K, evaluated at K. The main comparison.

| field | source | meaning |
|---|---|---|
| `experiment` | — | `"E1"` |
| `signature` | the fit | which checkpoint produced this. Stale when it no longer matches `summary.json` |
| `backbone`, `method`, `k`, `seed` | the cell | |
| `train_seconds`, `gpu_hours` | `summary.json` | carried along so a cost table needs only this file |
| `top1` | `knn.top1_accuracy` | percent of the 50 k validation queries whose first neighbour shares its label. The reported metric |
| `queries` | | number of queries scored |
| `empty_queries` | | queries whose code was entirely zero — see below |
| `seconds` | | the search loop only, between two device synchronisations |
| `backend` | | `blocked_fp32_gemm` |
| `mean_neighbour_score` | | mean cosine to the retrieved neighbour. A collapsing code shows up here as a score near 1 with falling accuracy |
| `nnz_mean`, `nnz_min` | the gallery encode | measured nonzeros per code, mean and minimum |
| `empty_gallery_codes` | | gallery rows with an all-zero code |
| `encode_seconds` | | gallery plus query encode time |
| `probe_top1`, `probe_top5` | `evaluate.probe` | linear probe on the full `hidden_dim` latent, 10 epochs, Adam, zero init, float32. Absent under `--no-probe` |
| `probe_settings` | | the probe's own settings block, including `feature: full_hidden_dim_latent` and `selection: fixed final epoch; validation labels only score` |
| `retrieval_ms`, `index_bytes`, `timing` | `evaluate.timing` | the papers' efficiency protocol (CSR App. E.3 / CSRv2 App. G.1): warm, resident, repeated batch search over a `torch.sparse_csr` gallery, excluding encoding, normalisation, transfers and index build. `timing` holds `cuda_seconds`/`wall_seconds` summaries, `queries_per_batch`, `milliseconds_per_batch`, `microseconds_per_query`, `queries_per_second`, the `index` block and `timing_scope` |

This is a secondary metric and the README records that the CUDA inverted index
in `csrc/` is not wired in; what is measured is the paper's `torch.sparse`
protocol.

### `eval_e2.json`

Only at a budget in the backbone's `e2_anchors`. One checkpoint, every inference
budget in the grid.

| field | meaning |
|---|---|
| `experiment` | `"E2"` |
| `signature` | the anchor checkpoint |
| `backbone`, `method`, `seed` | the cell |
| `anchor_k` | the K the checkpoint was *trained* at |
| `points` | one record per inference K, sorted by K |

Each point has the same evaluation fields as E1, including `probe_top1`,
`probe_epochs`, `probe_settings`, `timing_status`, `timing_config`, and measured
costs when requested/available. E2 honors the same probe and timing flags as E1.
At `k == anchor_k`, completed E1 metrics are copied rather than recomputed.
The enclosing `signature` identifies the frozen SAE for every point. Legacy
accuracy-only points are upgraded when probe/timing are requested; progress is
persisted after each component so interruption does not discard completed probes.
`cached_points` retains historical unselected K values outside the plotted sweep.

The file is rewritten after every budget, so an interrupted sweep resumes at the
first missing K.

### `failure.json`

Written by `run_grid` when a cell raises, immediately before the exception is
re-raised and the `train` stage stops.

| field | meaning |
|---|---|
| `cell` | `{backbone, method, k, seed}` |
| `error` | `"<ExceptionType>: <message>"` |
| `when` | local timestamp, `%Y-%m-%dT%H:%M:%S` |

A cell directory containing `failure.json` and no `eval_e1.json` is a cell that
was attempted and did not produce a number. The report counts it as incomplete.

## `runs/<backbone>/dense_reference.json`

The frozen backbone measured on its own, with no autoencoder: CSR Table 4's
"Full Rep." row. Written by `runs.dense_reference`, cached on the bundle
identity, and used both as the pipeline's correctness gate and as a numerical
diagnostic in tables; it is not drawn in the figures.

| field | meaning |
|---|---|
| `backbone`, `dim` | which embedding |
| `identity` | `canonical_hash(bundle.identity)` — the archive digests this measurement belongs to |
| `top1`, `correct`, `queries` | 1-NN top-1 of the raw embedding |
| `seconds`, `backend`, `gallery_rows` | as in E1 |
| `mean_neighbour_score` | mean cosine to the retrieved neighbour |
| `empty_queries` | zero for a dense embedding; a zero-norm row would have been rejected at extraction |
| `search_precision` | `float32_tf32_disabled` |
| `tie_rule` | `lowest_gallery_position` |
| `empty_query_policy` | `counted_and_scored_as_miss` |
| `representation` | `dense_backbone_embedding` |
| `index_bytes` | `train_rows × dim × 4` — the footprint the sparse codes are compared against on the efficiency figure |
| `published_top1`, `deviation` | only when the backbone has a published value. `ff2048` is the only one: 71.19 from CSR Table 4 |

## `runs/grid.json`

Written at the end of the `train` stage.

| field | meaning |
|---|---|
| `cells` | cells in the requested grid |
| `completed` | cells that returned a result in this invocation |
| `seconds` | wall-clock for the whole grid pass |
| `code_version` | `mpsae.__version__` |

`completed` counts this pass, not the tree: a resumed run that found everything
cached reports a small number of seconds and a `completed` equal to `cells`.

## `state/<stage>.json`

The receipts, from `logs.stage()`. One per stage name, plus one per backbone for
the embedding stage (`embed-ff2048.json` and so on).

On success:

| field | meaning |
|---|---|
| `ok` | `true` |
| `identity` | `canonical_hash` of the identity dict. Equality with the current invocation's identity is what makes a stage skip |
| `seconds` | how long the stage took |
| `finished` | local timestamp |
| `detail` | whatever the stage body put in its `detail` dict — for `env` the whole environment fact sheet, for `embed` the per-split archive manifests and the dense reference, for `ffcv` the written paths |

On failure the same file holds `ok: false`, `identity`, `error`
(`"<Type>: <message>"`), `seconds` and `finished`, and **no** `detail`. So after
an interruption this is the first file to read: it names the exception that
stopped the stage.

`state/options.json` is not a receipt. It records `{version, stages, options}`,
with every `PipelineOptions` field stringified, so the invocation that produced
a tree is recoverable. Credentials are not in `PipelineOptions` and never appear
here.

## Embedding archives

`embeddings/<backbone>/1K_{train,val}_<backbone>.npz` holds exactly two members,
uncompressed:

| key | dtype | shape |
|---|---|---|
| `data` | float32 | `[rows, dim]` |
| `labels` | int64 | `[rows]` |

The manifest travels with the archive as `<archive>.npz.json` — the receipt
keeps the file's own name plus `.json`, so `1K_train_ff2048.npz.json`. Written
by `embed.archive.finalize_archive`, which adds only what it can measure
itself; the recipe supplies the rest.

Measured by `finalize_archive`:

| field | meaning |
|---|---|
| `archive` | file name |
| `rows`, `dim` | shape of `data` |
| `keys` | `["data", "labels"]` |
| `dtypes` | `{data: float32, labels: int64}` |
| `precision` | `autocast_fp16_stored_float32` unless the recipe overrides it |
| `sha256` | digest of the NPZ. `features.load_bundle` refuses an archive whose bytes no longer hash to this |
| `labels_sha256` | digest of the label vector alone, so a relabelling is detectable without rehashing 10 GB |
| `deviations` | list of every deliberate difference from the released pipeline. `finalize_archive` appends the `label`→`labels` key rename unconditionally, because that rename is what makes CSRv2's supervised contrastive term live |
| `written` | local timestamp |
| `libraries` | versions of numpy, torch, torchvision, timm, ffcv, PIL, huggingface_hub, pyarrow — anything that can change a feature value |

Supplied by both recipes (and hashed into `identity`):

| field | meaning |
|---|---|
| `identity` | `canonical_hash` of the extraction identity. `load_current_archive` uses it to skip an archive that already matches |
| `recipe` | `mrl`, `csrv2`, or `synthetic-smoke` |
| `backbone`, `split`, `dim`, `input_size`, `batch_size` | the request |
| `normalize` | the `(mean, std)` actually used |
| `order`, `drop_last` | how rows were enumerated. This is why the two recipes have different row counts: the MRL recipe is sorted `ImageFolder` order with `drop_last=False`, the CSRv2 train pass is FFCV `OrderOption.RANDOM` with `drop_last=True` and so stops at 1,281,024 rows at batch 256 |
| `weights` / `pretrained_cfg` | provenance of the frozen model: for `ff2048` the pinned HF repo, commit and file digest; for timm the architecture, tag, `crop_pct`, interpolation, mean/std, `num_classes`, `hf_hub_id` |
| `recipe_reference` | the released file and function this reproduces |
| `preprocessing` | the full pixel path — resize, crop, interpolation, augmentation, loader order |
| `autocast`, `tf32` | `torch.autocast('cuda', float16)` and the observed TF32 flags |
| `seed` | the extraction seed, or `null` for the MRL recipe, where nothing is random |
| `imagenet` / `ffcv_dataset_manifest` | the source fingerprint: class and image counts plus a digest over sorted relative paths and sizes (mtimes excluded, so copying the dataset does not invalidate an archive) |
| `expected_dense_top1` | the published dense 1-NN this archive is gated against, or `null` |

A smoke archive carries `recipe: synthetic-smoke`, `source:
class-conditional Gaussians` and the deviation `SYNTHETIC DATA: smoke-test
archive, not an ImageNet embedding`.

## `train.jsonl`

One JSON object per line, one line per **logged** step. Logging is strided:
`FitSettings.log_every = 10`, plus an unconditional record at the final step. So
a 3,130-step FF2048 fit produces 313 records, not 3,130.

Always present:

| field | meaning |
|---|---|
| `wall` | `time.time()` when the record was appended — absolute, for correlating with other logs |
| `step` | optimiser step count after the update |
| `epoch`, `batch` | position in the schedule |
| `lr` | the preset learning rate. Constant by construction: the release builds `StepLR(step_size=10, gamma=1)`, which never changes it |
| `rows` | rows in that batch. The final batch of an epoch is short — `drop_last=False`, as released |
| `elapsed` | **cumulative** wall-clock since the fit began, including time from earlier resumptions |
| `training_k` | the budget actually used at this step. Equal to K for `csr` and `mpsae`; annealed only for `csrv2` |
| `view_2k`, `view_4k` | the other two gate budgets, `min(2k, h)` and `min(4k, h)` |
| `loss_mse_k`, `loss_mse_4k`, `loss_aux` | the three shared terms, **unweighted** — the weights (`0.125`, `1/32`) are applied in the total, not in these diagnostics |
| `loss_total` | the weighted sum actually backpropagated |
| `nnz_mean`, `nnz_min` | measured nonzeros in the main code |
| `below_k_fraction`, `empty_fraction` | share of rows with fewer than K nonzeros, and with none |
| `dead_fraction` | share of the dictionary currently past `dead_threshold`. The counter advances three times per step, once per gate, so `dead_threshold=30` is about ten optimiser steps |
| `active_units` | dictionary entries used by at least one row in the batch |

Method-dependent:

| field | present for |
|---|---|
| `loss_contrastive` | `csr`, `csrv2` — unweighted contrastive term; label-based only for `csrv2`, instance-based otherwise |
| `loss_ot` | `mpsae` — the unweighted matching gap |
| `ot_weight` | `mpsae` — multiplier of `loss_ot` in the total loss: 2.0 at training K ≤ 64; 0.5 above 64 |
| `ot_*` | `mpsae`: solver diagnostics averaged over the OT groups in the batch — `ot_mass`, `ot_mass_error`, `ot_cap_violation`, `ot_constraint_error`, `ot_iterations`, `ot_refinement_iterations`, `ot_objective_refinement`, `ot_negative_gap`, `ot_diagonal_plan_mass`, `ot_cost_mean`, `ot_cost_std_over_eta`, `ot_groups`, plus the worst case across groups in `ot_cost_min`, `ot_cost_max`, `ot_mass_error_max`, `ot_cap_violation_max` |

**`elapsed` is cumulative, not per-record.** Summing it across a file multiplies
the true total by the number of records. The total wall-clock of a fit is
`summary.json: train_seconds`; if that is missing, `max(elapsed)` is the
fallback, which is what `report.aggregate.training_summary` does.

## The reports tree

`report.build_all` writes two directories and returns a manifest listing every
table, every figure and every skip with its reason. Nothing in it raises on
missing data: a partially finished grid is the normal state of this project.

### `reports/tables/`

Each table is written twice. The CSV keeps `mean`, `std` and `n` in separate
columns at full precision, so a later script can recompute; the Markdown fuses
mean and std into one two-decimal cell, keeps `n` beside it, and carries a note
about what `±` means.

| table | one row per | contents |
|---|---|---|
| `e1_accuracy` | (backbone, method, K) | `top1`, `probe_top1`, `probe_top5`, `nnz`, `empty_queries`, `search_seconds`, `train_seconds`, `gpu_hours`, `index_bytes`, each as `_mean`/`_std`/`_n`, plus `seeds_present`, `seeds_missing`, `complete`, `code_versions` |
| `e2_generalisation` | (backbone, method, anchor K, inference K) | `top1`, `nnz`, `empty_queries`, `search_seconds` as above, plus `at_anchor`, `seeds_present`, `seeds_missing`, `complete` |
| `training_summary` | one fit | `logged_records`, `last_step`, `total_steps`, `finished`, `train_seconds`, `gpu_hours`, `step_seconds_mean`, `residency`, `final_*` for each loss and sparsity field, and `k_schedule` — realised budgets as `step:k` change points; only `csrv2` should anneal in current runs |
| `paired_<treatment>_vs_<control>` | (backbone, K) | per-seed differences for `mpsae` against each control in `csr`, `csrv2`: `delta_top1`, `delta_probe_top1`, `delta_nnz` as mean/std/n, plus `treatment_top1`, `control_top1` as context, and `seeds_paired`, `seeds_missing`, `seeds_unpaired` |
| `dense_reference` | backbone | our measured dense 1-NN as a numerical diagnostic, excluded from plots |

A metric nobody measured is dropped as a column rather than rendered as a column
of em-dashes — `probe_top5` with `--no-probe`, or `probe_top1` in the E2 table,
when probes are disabled for the sweep.

The downstream paired table compares MPSAE against CSR and CSRv2. Each
difference matches dataset/protocol, backbone, training K, inference K and seed,
and records both evaluation IDs. It is not a subtraction of unrelated averages.

### `reports/figures/`

Every figure is written three times: `<name>.png`, `<name>.pdf` from the same
figure object and the same bounding box, and `<name>.caption.txt` so the claim a
panel supports travels with the image instead of being re-invented in LaTeX.

| figure | supports |
|---|---|
| `backbone_summary` | the headline overview: accuracy vs K for every method, one panel per backbone, using only measured active sparse methods |
| `e1_curves_<backbone>` | compares MPSAE with CSR/CSRv2 at matched K; no published or dense reference curves |
| `e1_delta_<backbone>_vs_csr`, `e1_delta_<backbone>_vs_csrv2` | the same claim, paired per seed. This is the figure the headline sentence points at |
| `e2_generalisation_<backbone>` | linear-probe and 1-NN accuracy as inference K varies for each fixed training-K checkpoint |
| `e2_nnz_<backbone>` | where a larger nominal K stops buying real sparsity, plotted against measured nnz rather than the budget |
| `efficiency_<backbone>` | the trade-off CSR argues for, in the two units a deployment pays in: query wall-clock and index bytes. No band on the cost axes, because wall-clock spread is a property of the hardware, not the method |
| `training_<backbone>_k<K>` | internal diagnostic, not a paper figure: total loss, loss components, realised k-annealing and dead latents, one line per method per seed with no averaging, at the median trained K |

## Reading the numbers honestly

The default campaign uses seed **1227** only. Its curves have `n=1`, its
standard-deviation fields are empty, and no between-seed bands/error bars are
drawn. The multi-seed statistics below apply when multiple seeds are explicitly
selected. E2 panels hold training K fixed while sweeping the full inference-K
grid; hollow rings identify the matched training/inference K point.

**`±` is a sample standard deviation over seeds, with `ddof=1`.** It is a
spread, not a confidence interval, and not a standard error. At n=3 the
unbiased estimator differs from the population one by a factor 1.22, which is
large enough to matter in a paper, so `statistics.stdev` is used throughout and
`report.aggregate._stats` returns `None` rather than `0.0` below two
observations.

**`n` is always reported.** Every aggregate row carries `<metric>_n`; every
figure legend carries `(n=3)` or `(n=2–3)`; the Markdown tables keep an `n`
column, collapsed to one when every metric in a row rests on the same seeds and
expanded per metric exactly when they disagree. A mean without its `n` is not a
reportable number, and `seeds_missing` names which seeds are absent rather than
letting an unfinished cell look finished.

**There are no p-values anywhere.** Three paired observations cannot support an
inference claim, so none is made. Differences are reported as the mean and
sample std of the *per-seed* difference, never as a difference of means: a seed
fixes both the initialisation and the data order, so the same seed under two
methods is a matched pair and the difference cancels that nuisance variation. A
seed present for only one of the two methods is excluded from the pair, named in
`seeds_unpaired`, and logged — never silently dropped.

**Figures use measured active sparse methods only.** Published curves and
Dense/Full-representation overlays are disabled. `report.reference` is an
empty compatibility API. Our measured dense diagnostics remain available in
tables, separate from the CSR, CSRv2 and MPSAE curves.

### Nominal `k` versus measured `nnz_mean`

A code's budget is an upper bound on its nonzeros, not a count. The Top-K gate
selects K pre-activations and the ReLU then zeroes any that were negative, so a
code can hold far fewer than K nonzeros, and the gap widens with K. This is why:

* every E1 and E2 point carries `nnz_mean` and `nnz_min` beside its `top1`;
* `below_k_fraction` is recorded at encode time and `empty_fraction` during
  training;
* `e2_nnz_<backbone>` plots accuracy against the *measured* mean nnz, where a
  curve flattening marks the point at which raising K stops buying anything
  real.

Quoting a nominal budget as a sparsity figure overstates the real cost. Where
the two differ, `nnz_mean` is the honest number.

### `empty_queries` and `empty_gallery_codes`

An all-zero code. It scores 0.0 against every gallery vector, so the tie rule —
lowest gallery position, which is what both `torch.max` and `IndexFlatL2` give —
sends it to gallery row 0.

The policy, recorded in every search result as
`empty_query_policy: counted_and_scored_as_miss`: an empty query is **counted in
the denominator and scored as a miss**. An empty code carries no information, so
crediting it with whatever neighbour the tie-break happened to return would
inflate exactly the methods whose codes collapse — the failure mode most worth
seeing. Only all-zero codes are forced to miss; a nonempty code that happens to
share no latent with any gallery row also scores 0.0, lands on row 0, and is
left to that outcome, as it would be in FAISS.

`empty_gallery_codes` is the same count on the gallery side. It is a diagnostic,
not a correction: an empty gallery row stays in the index and can be retrieved
by an equally empty query.

## Loading results yourself

`mpsae.report.aggregate` is pure standard library — no numpy, no torch — so this
runs on a laptop against a copied `work/` tree.

```python
from mpsae.report.aggregate import aggregate_e1, load_runs, paired_difference

records = load_runs("work")                 # or "work/runs"; both are accepted
print(sum(1 for r in records if r["complete"]), "of", len(records), "runs have E1")

for row in aggregate_e1(records):
    if row["backbone"] == "ff2048" and row["k"] == 32:
        print(f"{row['method']:14s} "
              f"top1 {row['top1_mean']:.2f} ± {row['top1_std'] or float('nan'):.2f} "
              f"(n={row['top1_n']}, seeds {row['seeds_present']}"
              + (f", missing {row['seeds_missing']}" if row["seeds_missing"] else "")
              + f") nnz {row['nnz_mean']:.1f}")

for row in paired_difference(records, treatment="mpsae", control="csr"):
    if row["backbone"] == "ff2048":
        print(f"K={row['k']:5d} Δtop1 {row['delta_top1_mean']:+.2f} "
              f"± {row['delta_top1_std'] or float('nan'):.2f} "
              f"over seeds {row['seeds_paired']}")
```

A few things that are easy to get wrong when reading this tree directly:

* `row["top1_std"]` is `None`, not `0.0`, when only one seed is present. Check
  `top1_n` before formatting.
* `aggregate_e2` keys on `(backbone, method, anchor_k, k)` and sets
  `at_anchor = (k == anchor_k)`. The anchor point is the E1 measurement copied
  across, so it carries fewer fields than the swept points.
* `load_runs` warns about run directories outside the configured grid but still
  returns them. If you narrowed `--backbones` or `--methods` since an earlier
  pass, those older cells are in the records.
* `training_summary` reads the tree itself rather than taking the records, and
  down-samples each `train.jsonl` to at most 400 curve points under
  `row["curve"]`. The `curve` key is excluded from every table; it exists so the
  diagnostic figure and the table are backed by the same numbers.

## Downstream DINO kNN result records

The campaign's downstream records are distinct from ImageNet `eval_e1.json` and
`eval_e2.json`. EuroSAT records use `task: knn_classification`, store the DINO
weighted-vote classifier settings, and include `knn_<n>_top1`, `knn_<n>_top5`,
`knn_<n>_balanced_accuracy` and `knn_<n>_macro_f1` for each configured neighbour
count. Defaults are n = 1 and 20 with temperature 0.07. `eval_k` remains the sparse
embedding budget; it is not the number of neighbours.

One record represents one dataset/backbone/checkpoint/inference-K condition and
contains both classifier results. The search is shared across neighbour counts;
search seconds and voting seconds describe separate work. Classification tables
and figures show the voted class metrics, while SOP retrieval records retain
their Recall/Precision/mAP fields. No retired methods or dense/published reference
curves are drawn. Dense diagnostic values remain available numerically.

The evaluation identity includes the task and classifier recipe, so changing
temperature, neighbour counts or edge-case policy cannot reuse old metrics.
Changing only the downstream classifier does not change the source checkpoint
identity or require another SAE fit. See [the benchmark protocol](benchmark.md)
for support/query splitting, exact vote semantics, deterministic ties and
zero-code abstention.
