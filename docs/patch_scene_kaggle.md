# Kaggle: 24-point patch SAE benchmark

Open `notebooks/patch_scene_kaggle.ipynb` on Kaggle. Push this code to the
`refactor` branch before running the notebook. Enable Internet and select a
T4 GPU. **No Hugging Face account, token, secret, or artifact repository is
required.** The notebook downloads public models/datasets but keeps all
experiment outputs on Kaggle. No upload flags are passed.

If you already have an open Kaggle notebook, use **File -> Import Notebook**
and upload `notebooks/patch_scene_kaggle.ipynb`, then run its cells in order.
Start with `ENCODER = "dinov2"` and `DATASET = "voc"`.

The notebook creates an isolated Python 3.11 environment with pinned packages
and CUDA 12.6 PyTorch wheels, independently of Kaggle's notebook Python.
It does not install drivers or use the old server setup script. The current
GPU guard does not support P100. T4 x2 uses GPU 0 only; VRAM is not pooled.

## Select a job

Set `ENCODER` to `dinov2`, `siglip2`, or `clip`; set `DATASET` to `voc` or
`ade20k`. Start with DINOv2/VOC. Each job performs three fresh patch SAE fits
and four validation evaluations (dense, CSR, CSRv2, MPSAE). Six jobs complete
the campaign. Run one job at a time initially; do not assume the entire
campaign fits one Kaggle session. `all` is supported for sequential runs.

The original image encoder is frozen. These are not the previously disabled
backbone fine-tuning jobs. No ImageNet download or old SAE checkpoint is needed.

```bash
python -m mpsae.patch_scene plan --config configs/patch_scene_kaggle.yaml
python -m mpsae.patch_scene run --config configs/patch_scene_kaggle.yaml \
  --encoder dinov2 --dataset voc --device cuda:0 \
  --workspace /kaggle/working/patch-scene-v2
```

Use `/tmp/mpsae-env/bin/python` for these commands in the supplied notebook.
The default run saves locally. Optional CLI flags `--repo` and `--publish`
still support remote restoration/backup for users who explicitly request it;
they are not used by the Kaggle notebook.

## Fixed experiment protocol

| Setting | Value |
|---|---|
| Encoders | SigLIP 2-B/16, DINOv2-B/14, OpenAI CLIP-B/16 |
| Input / SAE latent dimensions | 768 / 8192 |
| Target train K / inference K | 12 / 12 |
| SAE training | 10 epochs, batch 1024, Adam LR 4e-5 |
| Training / support bank limits | 262144 / 65536 patches |
| Search | exact FP32 cosine, 30 neighbors, temperature 0.02 |
| Input geometry | full image resized to 448 square; no center crop |
| Seed | 1227 |
| Output | CSV, JSON, LaTeX table; no figures |

The registered methods' losses are reused, including CSRv2's training-K
annealing and MPSAE's MMPOT term. All methods use the same fresh initialization,
sampled features and batch ordering. The campaign explicitly uses contrastive
coefficient 0.1 and MMPOT coefficient 2.0 at target K=12. It does not change
the old backbone-specific presets. It retains the existing OT convergence
settings; timing-loop repetition settings are unrelated to OT solver accuracy.

SAEs train on the selected dataset's training split only, starting from random
SAE weights. Validation images never enter SAE fitting. CSRv2 reads the dominant
nonvoid training-mask class; CSR/MPSAE do not use class targets in their loss.
The support sampler uses training labels for every method. Report CSRv2's extra
supervision. Train on VOC and ADE separately; this is dataset-specific patch
adaptation, not cross-dataset SAE transfer or Hummingbird pretraining reproduction.

SigLIP/CLIP produce 28x28 patch grids; DINOv2 produces 32x32. CLIP uses the
vision model's post-layernorm applied to spatial tokens, not its 512-dimensional
pooled projection. CLS is removed from CLIP/DINOv2; the chosen DINOv2 has no
registers. SigLIP has no CLS. Positional interpolation is enabled as needed.
Pixel normalization comes from each pinned checkpoint's processor.

Training patches are sampled without replacement per image, uniformly over
nonvoid patches. Some images can supply fewer patches than their quota. The
support bank is sampled once from the training bank using the existing
label-informed sampler and shared across all four evaluations.

## Efficiency and memory

The encoder is loaded once for extraction, then released. Extracted features
are shared by all methods. Default FP16 storage has a checked relative squared
rounding error bound; training and search use FP32. This is not a full FP32
extraction-equivalence claim. Set `cache_dtype: float32` for that ablation;
changing the config creates a separate campaign and cache identity.

Search stores compact K-sparse codes but builds a dense transposed lookup as
a GPU workspace: about 2 GiB for 65536 x 8192 float32 values. It scores only
the 12 query coordinates, avoiding full 8192-dimensional query GEMMs. This is
exact exhaustive search, not ANN. Equal scores use ascending gallery indices,
including ties at the neighbor cutoff. The workspace is reported separately;
do not claim the whole search index occupies only compact sparse-code storage.
Native-size class probabilities are interpolated in row tiles to bound memory.

The notebook places downloaded datasets/model caches under `/tmp`, outside
saved Kaggle outputs. Only selected feature shards and run artifacts are under
`/kaggle/working`. Monitor disk space; six jobs' caches may exceed saved-output
limits. Feature shards are reused locally while the workspace remains.
There is no remote backup in the default notebook.

The query timing field includes streaming, encoding, search, upsampling,
metrics and periodic backups. It is not a kernel-only retrieval latency.
Profile actual VRAM/time on the pilot; no GPU runtime or memory guarantee is
implied by CPU tests. The backbone batch size is independent of SAE batch size.

## Download results directly from Kaggle

After each completed evaluation, the runner atomically refreshes:

```
/kaggle/working/patch-scene-v2/patch-scene-results.zip
```

The notebook also copies it to `/kaggle/working/patch-scene-results.zip` when
the training cell exits (including a Python exception), and displays a download
link in its final cell. In Kaggle's file browser, find that ZIP under
`/kaggle/working` and select **Download**. It is available with partial results;
you do not need to wait for all 24 evaluations.

Contents: `results.csv`, `results.tex`, `results.json`, `config.json`,
`sources.json` when available, and `README.txt`. The ZIP excludes feature
caches and model/optimizer checkpoints to keep the download small. It is a
results archive, not a training-resume backup.

To rebuild the ZIP from already completed local evaluations, without training,
GPU use, or any artifact-service connection:

```bash
python -m mpsae.patch_scene export --workspace /kaggle/working/patch-scene-v2
```

Tables live at:

```
<workspace>/campaigns/<campaign-id>/table/results.csv
<workspace>/campaigns/<campaign-id>/table/results.tex
<workspace>/campaigns/<campaign-id>/table/results.json
```

Missing cells are `--`, never zero. The table becomes complete at 24 entries.
Per-cell artifacts avoid overwriting other encoder/dataset jobs. Jobs run in
the same retained workspace are combined automatically. Results from separate
sessions whose workspaces were discarded are not automatically combined.

## Local resume and session lifetime

Training saves locally every 128 steps and at epoch boundaries. Extraction
commits every completed shard; evaluation commits progress every 64 images
and at completion. Rerun the same command/config to resume from those files
as long as the workspace remains. Do not change package versions/configuration
if you intend to reuse exactly the same experiment.

Download the results ZIP before ending the session, or save a Kaggle notebook
version **including output files**. Saving only notebook code does not preserve
results. A reset/deleted session without retained outputs loses its local
checkpoints/cache; this workflow makes no remote recovery promise. To resume
training in another session, preserve/restore the full workspace, not just the ZIP.

Optional remote backup is still available outside this notebook:

```bash
python -m mpsae.patch_scene run --encoder dinov2 --dataset voc \
  --workspace work/patch-scene-v2 --repo YOUR_ACCOUNT/YOUR_DATASET_REPO --publish
```

For an offline synthetic end-to-end check (not benchmark results):

```bash
python -m mpsae.patch_scene smoke --device cpu --workspace work/patch-smoke
```

Smoke runs are always under `SMOKE_ONLY`; publishing them is prohibited.
The main CSV contains mIoU/pixel accuracy in percent and measured mean nonzeros.
The LaTeX table contains mIoU only. Single-seed results have no error bars.
