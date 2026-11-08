# Data

Where the pixels come from, how each backbone's embeddings are extracted, and
what is recorded so a number can be traced back to its inputs months later.

Nothing is reused from the previous pipeline. The published archives are not
read, because no released code produces the FF2048 ones (see below) and the
label key in them makes CSRv2's supervised loss inert.

## ImageNet-1K

Source: Hugging Face `ILSVRC/imagenet-1k`. The repository is **gated per
user** — accept the terms once with the account that owns `HF_TOKEN`, or every
download fails with a message naming that URL.

The stage resolves `main` to a concrete commit and pins it. Re-running months
later therefore fetches the same bytes, and the revision is recorded in every
downstream manifest.

### Label to synset

The parquet `label` column is a bare integer and the `ClassLabel` names are the
human-readable descriptions, so neither gives the synset. The mapping comes
from `classes.py` in the pinned revision, whose `IMAGENET2012_CLASSES` is keyed
by synset in label order. It is parsed with `ast.literal_eval` — read as a
literal, never executed.

Three checks, because an off-by-one here would silently mislabel everything:

* exactly 1000 keys, all matching `n[0-9]{8}`;
* the keys are in sorted order, since `ImageFolder` sorts its class directories
  and the two label spaces must agree;
* for every train row, the synset prefix of the original file name matches the
  synset derived from its label. Train files are named `n01440764_10026.JPEG`,
  so this is a free end-to-end check of the mapping.

### Materialising

Both downstream recipes read pixels rather than parquet — the MRL recipe uses
`torchvision.datasets.ImageFolder`, the CSRv2 recipe uses an FFCV file written
from the same folders. So each shard is copied out to
`work/imagenet/{train,val}/n########/*.JPEG` as the **original JPEG bytes**.
Never a re-encode: a second JPEG pass changes the pixels that both published
recipes were measured on.

Each parquet shard is deleted once written, so peak disk is one shard plus the
images rather than 155 GB twice over.

Resume is per shard. A shard is recorded only after `os.sync()`, and an
interrupted shard is rewritten from the start — a half-written shard is never
trusted. Changing the pinned revision invalidates the receipt and the stage
refuses to continue rather than mixing releases.

The split is accepted only when all of these hold:

* 1,281,167 train rows and 50,000 validation rows;
* all 1000 classes received images;
* the count on disk matches the receipt;
* validation is exactly 50 images per class — if it is not, the label mapping
  or the shard set is wrong.

### FFCV

The CSRv2 recipe needs its input serialised, with the parameters from the CSRv2
README: `max_resolution=500`, `write_mode=proportion`,
`compress_probability=0.50`, `jpeg_quality=90`. This is a straight port of
their `dataset_preparation/write_imagenet.py`.

FFCV is the one awkward dependency; it is needed by four of the five backbones
and not by `ff2048`. See [`server.md`](server.md).

## Embedding extraction

Two recipes, because the two backbone families were measured with two different
pipelines and following only one of them would break a comparison.

### ff2048 — the MRL recipe

FF2048 is MRL's **F**ixed **F**eature ResNet-50: trained with the same 40-epoch
FFCV recipe as their Matryoshka models but without the nesting, giving a plain
2048-d embedding. CSR uses it so that CSR-vs-MRL is a comparison on one
backbone, and its dense 1-NN of 71.19 is the "Full Rep." row of CSR Table 4.

**No released code produces these embeddings.** CSR's extraction script only
handles timm models, and FF2048 is not one. Two pieces of evidence say the
published archive came from MRL's own retrieval pipeline instead: the file
naming follows MRL's `1K_train_mrl0_e0_ff2048` convention, and the row count is
exactly 1,281,167, which is incompatible with the CSR script's `drop_last=True`.

So we follow MRL's `inference/pytorch_inference.py --retrieval`:

* weights from `aniketr/mrl-resnet50`,
  `fixed-feature/Imagenet1k_R50_ff2048/final_weights.pt`, pinned and hashed;
* `torchvision.models.resnet50` with BlurPool applied, loaded `strict=True`
  after stripping DDP/Sequential prefixes, `fc = Identity()`;
* `Resize(256)` → `CenterCrop(224)` → ImageNet mean/std, the **same transform
  for the database and the query set** — no augmentation on either;
* `shuffle=False`, `drop_last=False`, so every image is kept in source order;
* the 2048-d avgpool output under fp16 autocast, stored float32.

Two deviations are recorded, both about storage rather than pixels: labels are
stored `int64` rather than MRL's `float16` (exact over 0..999, but a float
label key is a trap for the contrastive loss), and each split is one NPZ rather
than separate `-X.npy`/`-y.npy` files, so features and labels cannot be paired
up in the wrong order.

**The gate.** Before any training, the dense 1-NN of these embeddings must
reproduce 71.19. A deviation above 0.3 points stops the pipeline: it would mean
the recipe diverged from MRL's, and every comparison against the published
tables would be meaningless. A smaller deviation is logged and recorded in the
manifest.

### The timm backbones — the CSRv2 recipe

`rn18`, `convnext_tiny`, `vit_b16`, `dinov2_vits14` all go through
`CSRv2/image/pretrained_embeddings.py`:

* `timm.create_model(tag, pretrained=True)`, pinned revision, weight digest
  recorded; the feature is `forward_head(forward_features(x), pre_logits=True)`;
* **train**: `RandomResizedCrop(224)` + horizontal flip, random order,
  `drop_last=True`;
* **validation**: centre crop at ratio 224/256, sequential, `drop_last=False`;
* fp16 autocast, stored float32.

Note what this means: the train embeddings are a **single augmented pass**, so
they are a random draw, and `drop_last=True` discards the last partial batch —
1,281,024 rows rather than 1,281,167. That is the released recipe and we keep
it; we only pin and record the loader seed, which the released script left
unset, so the draw is reproducible.

#### Deviations

Recorded in the archive manifest with the reason each was forced:

1. **The validation crop is `input_size` (224), not the hard-coded 256.** timm's
   ViT-B/16 is fixed at 224 and DINOv2 uses patch 14, so 256 either errors or
   silently changes the model.
2. **The loader seed is pinned** and recorded; the released script set none.
3. **Mean/std come from the model's `pretrained_cfg`** rather than the
   hard-coded 0.5 — but only where they actually differ. ViT augreg really was
   trained at 0.5, so it records nothing here; ConvNeXt `fb_in1k` and DINOv2
   were trained with ImageNet statistics and do. `rn18` pins 0.5 in the registry
   and so records nothing either.

Separately, three fixes to the released plumbing, which are bugs rather than
algorithm and are not listed as deviations: batches are concatenated in numeric
order rather than `os.listdir` order; labels are carried alongside their
features instead of being dropped entirely; and the output is a single NPZ
rather than loose `.npy` files.

## Archives

One NPZ pair per backbone, plus a manifest:

```
work/embeddings/<backbone>/1K_train_<backbone>.npz    {data: float32[n,d], labels: int64[n]}
work/embeddings/<backbone>/1K_val_<backbone>.npz
work/embeddings/<backbone>/*.json                      manifest
```

The key is **`labels`**, plural. Every released producer writes `label` while
the released CSRv2 loader reads `labels`, so with the published files every
label becomes 0 and the supervised contrastive term collapses to ≈0. Renaming
it is a deliberate fix, recorded in the manifest's `deviations` list, and
`mpsae/features.py` warns loudly if it ever loads an archive carrying the old
key.

The manifest records the recipe, the split, rows and dimension, the source
weights with repository, revision and digest, the ImageNet provenance, the
exact preprocessing in words, the precision, the seed, the library versions,
the label digest, the archive digest and the deviation list. The digest is
verified on every load, and the memory-mapped cache is keyed by it.

Approximate sizes: ff2048 10.5 GB train, the 768-d backbones 3.9 GB, rn18 2.6
GB, dinov2 2.0 GB; validation splits are 1/25th of those.

## Embeddings repository (built once, then pulled)

`python -m mpsae.embeddings_hub` (also `bash embeddings.sh`) keeps every
embedding set in one folder `<dataset>_<backbone>_<dim>` — `imagenet_ff2048_2048`,
`sop_rn18_512`, `eurosat_dinov2_vits14_384` — locally under
`work/embeddings-hub/` and in one private HF dataset, `Lukita/MPSAE-embeddings`.

* Per folder: on this machine and on the Hub -> use the local one; only on
  this machine -> push it; only on the Hub -> pull it (every SHA-256 checked);
  nowhere -> build it with the campaign's own extractors (`extract_source`,
  `extract_bundle`; `storage_mode: direct_v1`), push it, verify the remote
  hashes, and only then delete the raw data. Pushes check the local hashes
  first and the remote hashes after.
* Repository: `Lukita/MPSAE-embeddings` (`--repo owner/name` to change); `.env`
  only needs `HF_TOKEN` with access to it. Created private if missing.
* Downstream features use the backbone weights recorded in that backbone's
  ImageNet folder, so ImageNet is always done first.
* `bash all_in_one.sh` runs `embeddings_hub sync --config <campaign>` before the
  campaign (`--skip-embeddings` turns it off) and hard-links the folders where
  the campaign reads them: ImageNet archives into
  `work/direct-v1/embeddings/<backbone>/`, downstream folders into
  `work/direct-v1/benchmarks/<dataset>/hub/<backbone>/`. The campaign still runs
  its own verification and uses a downstream folder only when its weights equal
  the source checkpoint's; otherwise it extracts as before.
* The campaign's own backup then uploads these features again to `<account>/MPSAE`
  (its artifact backup); that copy is redundant but harmless.

## Publishing

Optional, and never implied by a bare run: the publish stage is skipped unless
`HF_EMBEDDINGS_REPO` is set, which is what authorises the upload. It creates a
**private** dataset repository, because ImageNet-derived features inherit the
dataset's non-commercial terms, and pushes one folder per backbone with the
manifests alongside.

Once the archives exist, `work/imagenet/` and `work/ffcv/` can be deleted —
about 355 GB back. Nothing downstream reads them.

## Splits

Training uses the whole train split as both the fitting set and the retrieval
gallery, and the 50k validation split as the query set. That is the protocol of
CSR Table 4 and of `CSRv2/image/faiss_nn.py`.

Nothing is held back. The previous pipeline reserved 1,024 rows as "anchors"
for a landmark variant that no longer exists; keeping that reservation would
have shrunk the gallery and made the dense gate unable to reach 71.19.

## Downstream benchmarks

`data_usage.md` lists the datasets for the next phase — Flowers-102, DTD,
VOC 2012, SPair-71k, SOP, DeepFashion In-Shop, ROxford/RParis. None are
implemented yet. Two notes for when they are:

* that file's DTD test split should be 40 images per class, not 10;
* the official Oxford/Paris image archives have been withdrawn from the VGG
  pages, so the HF mirror is the practical route, and DeepFashion In-Shop needs
  a signed agreement.
