# One-file server setup: segmentation mass and weight comparisons

Copy `scripts/segmentation_server_onefile.py` to a new Linux NVIDIA server and
run it with Python 3. Python 3, git, curl, bash and the NVIDIA driver must already
be available (use the provider's NVIDIA/PyTorch image). No Conda, tmux, manual
dataset upload, Hugging Face token or result upload is required.

Push the latest code to the GitHub `refactor` branch first, then run:

```bash
curl -fL https://raw.githubusercontent.com/thienquang204/MPSAE/refactor/scripts/segmentation_server_onefile.py -o run_segmentation.py
python3 -u run_segmentation.py --workspace /workspace/segmentation
```

Defaults match the Kaggle mass/weight example: VOC, SigLIP2, CSR/CSRv2/MPSAE,
train/evaluate K=3/12/48, s=.9/.95/1, lambda=.5/1/2, dense included. This is
33 unique fits and 34 unique evaluations. The script clones the repository,
uses its pinned Python 3.11 setup, downloads data, and runs independent jobs on
up to two visible GPUs. Installation is reused unless the pinned requirements
or setup script change. It never silently updates an existing checkout; use
`--update` for an explicit fast-forward pull.

Example overrides:

```bash
python3 -u run_segmentation.py --workspace /workspace/segmentation \
  --dataset nyuv2 --encoders siglip2 dinov2 --methods csr csrv2 mpsae \
  --train-k 3 12 48 --masses 0.5 0.7 0.9 0.95 1 --weights 0.5 1 2
# For fewer runs: --encoders siglip2 --masses 0.9 1 --weights 2
# For only MPSAE: --methods mpsae --no-dense
# To choose GPUs: --devices cuda:0 (or --devices cuda:0 cuda:1)
```

Data preparation happens once. VOC is fetched by the existing feature loader;
NYUv2 and Cityscapes use their existing preparation modules. No training or
evaluation algorithms are duplicated. Partial transport and balanced MMOT use
the same solvers as Kaggle. Dataset-specific epoch settings are preserved.

Results:

```text
/workspace/segmentation/segmentation-results.zip
/workspace/segmentation/results/<experiment_id>/
    plan.json
    results.json
    results.csv
    mass_weight_comparison.csv
    segmentation-results.zip
```

The ZIP includes all selected masses and weights, combined CSVs and each sweep's
LaTeX tables/provenance ZIP. Shared baselines are deduplicated and checked for
consistency. An interrupted/failed run exports available results and reports
missing conditions in results.json; missing scores are never zero. Historical
experiments retain distinct output folders. The top-level ZIP is a convenient
copy of the latest invocation's results.

Repeat the same command/workspace to resume compatible fits and evaluations.
Model checkpoints, images and feature caches are excluded from the results ZIP;
retain the workspace to resume. The script runs in the foreground. Keep the
terminal connected while it runs.

`--plan` prints resolved settings without preparing data/training, but a new
server still needs environment setup first. For an already installed Python
environment, use `--skip-setup --repo /path/to/MPSAE`; this also makes planning
independent of NVIDIA availability. `--smoke` uses isolated synthetic data and
is never a real benchmark. No concurrent launchers in the same workspace.

Local checks cover planning, installation reuse, partial/balanced CPU training,
two-worker execution, aggregate exports and checkpoint reuse. A fresh Linux
CUDA installation and full dataset benchmark require validation on the server.
