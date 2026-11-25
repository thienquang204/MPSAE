# CVPR 2027 experiment reproduction

This folder organizes the experiments in the 18-page `CVPR_2027_MPSAE.pdf`
supplied on 2026-11-25. The PDF fingerprint is in `reference/paper.sha256`.
The source checkout inspected for this inventory was `e504eb7`; this is not a
claim that every historical result was generated at that commit.

The shared numerical implementation remains in `mpsae/`. Existing configs,
launchers, checkpoint formats and workspaces remain supported. This folder is
the paper-facing catalog and reporting interface. Cityscapes and linear-probe
**segmentation** are documented in `../extensions/`; they are outside this draft.
ImageNet linear probing **is** part of this paper.

## Start with existing results

Run these commands from the repository root. Reporting needs no PyTorch, GPU,
dataset download, Hugging Face account or token:

```bash
python -m pip install -r experiments/cvpr2027/environment/report.txt
python experiments/cvpr2027/reproduce.py plan
python experiments/cvpr2027/reproduce.py index /path/to/mpsae-results.tar.gz
python experiments/cvpr2027/reproduce.py index /path/to/patch-scene-results.zip
python experiments/cvpr2027/reproduce.py index /path/to/retrieval-studies
python experiments/cvpr2027/reproduce.py report --tables 1 12 13
```

`index` accepts an extracted workspace, ZIP or tar archive. Repeat it for every
saved campaign; exact duplicate receipts are deduplicated. Nothing is extracted
over your repository, no `.pt` file is loaded, and original files are not moved.
The original JSON receipts are copied into content-addressed `imports/` folders.
CSV-only summaries cannot establish training provenance; retrieve the original
`eval_e1.json` / `eval_e2.json` and fit metadata, or segmentation `results.json`
plus `config.json`. Unsupported or unmatched records are listed in each import's
`excluded.json`, rather than assigned guessed hyperparameters.

Results go to `work/cvpr2027/` by default. Pass `--output /path/to/output` to
every command to use another location. To inspect incomplete imported results:

```bash
python experiments/cvpr2027/reproduce.py report --allow-partial
python experiments/cvpr2027/reproduce.py verify
python experiments/cvpr2027/reproduce.py bundle
```

The ZIP is `work/cvpr2027/cvpr2027-results.zip`. It contains original metadata,
the index, recipes, tables and figures, with no weights or feature matrices.
Download it directly from your server or Kaggle output. Restore the ZIP by
extracting it to a new **output directory**, then invoke `report --output ...`
from the matching repository checkout. No result uploads are performed.

`report` requires all selected table conditions by default. `--allow-partial`
marks incomplete tables and leaves missing values as `--`; it does not use PDF
values to fill holes. `verify` exits nonzero for missing/conflicting conditions
or discrepancies against the checked-in **spot-check** reference cells. The
reference CSV is not a complete transcription of all paper numbers.

## Run on a fresh server

The existing combined setup installs the CUDA environment used by retrieval and
patch segmentation. From an already cloned repository:

```bash
bash scripts/setup_ablation79.sh
source .venv-retrieval/bin/activate
cp experiments/cvpr2027/paths.example.yaml experiments/cvpr2027/paths.local.yaml
```

Edit `paths.local.yaml` to point at existing workspaces if available. Paths in
that file resolve relative to the file, not your shell's current directory.
`imagenet` points at `direct-v1` itself; `retrieval` and `segmentation` point at
the roots containing their `fits/`, `sweeps/` or `campaigns/` directories.
The example works without edits for fresh repository-local workspaces.

```bash
# Inspect exact job counts first; no GPU or downloads.
python experiments/cvpr2027/reproduce.py plan --studies retrieval_mass

# Train missing fits and evaluate; cache global ImageNet embeddings once.
python experiments/cvpr2027/reproduce.py run \
  --studies retrieval_mass retrieval_weight retrieval_k2 \
  --paths experiments/cvpr2027/paths.local.yaml --download-embeddings

# Main ImageNet includes its 84 supervised linear probes.
python experiments/cvpr2027/reproduce.py run --studies imagenet_main \
  --paths experiments/cvpr2027/paths.local.yaml --download-embeddings

# VOC or NYUv2, using fresh patch SAEs on frozen encoders.
python experiments/cvpr2027/reproduce.py run --studies voc_main voc_weight voc_mass voc_joint \
  --paths experiments/cvpr2027/paths.local.yaml
python experiments/cvpr2027/reproduce.py run --studies nyuv2_main nyuv2_weight nyuv2_joint \
  --paths experiments/cvpr2027/paths.local.yaml
```

NYUv2 is downloaded through the existing pinned labeled-MAT helper unless
`nyuv2_root` points to an already prepared directory. VOC uses the existing
dataset loader. `--device cuda:0` and `--cpu-threads 4` are configurable.
The paper frontend schedules jobs sequentially on one GPU; existing Kaggle
multi-GPU launchers remain available. Do not run two writers on one workspace.
Changing hardware requires recording the actual environment; exact bitwise
agreement across GPUs is not assumed.

For existing compatible checkpoints, use `evaluate` instead of `run`. This mode
**cannot take SAE optimizer steps**, including resuming incomplete fits. Missing
or incompatible checkpoints cause an error. ImageNet linear-probe classifiers
may still be fitted when their requested evaluation is missing. Re-extracting
features may be necessary if their caches are absent.

```bash
python experiments/cvpr2027/reproduce.py evaluate --studies voc_main \
  --paths experiments/cvpr2027/paths.local.yaml
```

Every successful run, and indexable partial work after failure, is indexed in
the selected output directory. Reports are rebuilt explicitly with `report`.
Individual original runners also continue to produce their usual reports.

## Catalog and paper mapping

| Study | SAE fits | Evaluation conditions | Paper |
|---|---:|---:|---|
| `imagenet_main` | 39 | 222 1-NN + 84 probes | Table 1, Figure 2, Tables 12–13 |
| `retrieval_mass` | 16 | 48 | Table 3 |
| `retrieval_weight` | 10 | 20 | Table 4 |
| `retrieval_k2` | 2 | 2 | Table 5, plus four reused Table 3 results |
| `voc_main` | 18 | 54 sparse + 2 dense | Table 2 |
| `nyuv2_main` | 18 | 18 sparse + 2 dense | Table 2 |
| `voc_weight` | 12 | 36 | Table 2; default weight reuses `voc_main` |
| `nyuv2_weight` | 18 | 18 | Tables 2 and 10 |
| `voc_mass` | 36 | 108 | Tables 6–7 |
| `voc_joint` | 72 | 72 | Tables 8–9 |
| `nyuv2_joint` | 27 | 27 | Table 11 |

After overlap removal: **238 SAE fits, 84 linear-head fits, 671 metric
evaluations** (292 ImageNet 1-NN, 84 ImageNet probes, 295 segmentation including
four dense references). These are logical requirements of the entire draft,
not a claim that their artifacts are present and not 671 new runs to launch.
The older 79-evaluation launcher remains a separate suite.

## Files and provenance

- `configs/`: explicit paper recipes. `grid` and `variants` expand through the
  existing configuration classes, without copying training algorithms.
- `reference/recipes.lock.json`: resolved preset/condition snapshot; detects
  numerical default drift before launch. `source.lock.json` checks scientific
  implementation files. Locks describe this reproduction recipe, not recovered
  historical model/dataset revisions.
- `paper_map.yaml`: explicit table membership and Figure 2 selection. Figure 1
  is an illustration; its editable source has not been supplied.
- `results/metrics.csv`: one row per logical evaluation, with train/inference K,
  mass, loss weight, batch, metric, status and measured sparsity.
- `results/provenance.jsonl`: candidate records with original receipt paths,
  checkpoint/training identities and available evaluation metadata.
- `reports/tables/table_XX.csv` / `.tex`: exact selected rows and formatted tables.
  Compile `reports/tables/preview.tex` with a normal LaTeX installation.
- `reports/figures/`: Figure 2 and sensitivity plots, vector PDF and 300-dpi PNG.
  Missing curves remain gaps; plots are omitted when no real scores are present.
- `launches/`: requested config, code revision, environment and run status.

Two different candidate results for the same condition cause a conflict.
Inspect `results/index.json`, then record an explicit choice, never a best-score
or latest-timestamp selection:

```bash
python experiments/cvpr2027/reproduce.py select \
  --condition CONDITION_ID --candidate CANDIDATE_ID
```

Preserve the whole original workspace to resume training. A result-only ZIP
cannot recreate missing checkpoint weights. “Indexed” means that recorded
settings matched this catalog; it does not mean independent GPU reproduction.
NYUv2 uses the official test split: report sensitivity, not a test-selected
winning hyperparameter as an unbiased tuned result.

## Maintaining and extending experiments

Add a recipe and a named catalog entry with expected counts, then map its
outputs. Keep experiments outside this draft in `../extensions/` or a new paper
directory. For an intentional change to this frozen paper recipe, review and
regenerate the locks with `freeze.py --write` in the same commit; do not edit
locks just to bypass a failed check. Preserve historical configurations and
source identities. Formatting-only changes do not alter fit/evaluation IDs.

Implementation checks and known existing test failures are recorded in
[VALIDATION.md](VALIDATION.md).

The source dataset and encoder revisions behind the historical PDF still need
to be recovered from original receipts. See `reference/discrepancies.md` before
describing a fresh run as an exact historical reproduction.
