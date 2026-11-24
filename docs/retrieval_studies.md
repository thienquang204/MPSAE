# MPSAE retrieval ablations

The current `configs/retrieval_studies.yaml` defines **28 ImageNet SAE fits and
72 evaluations**: 48 mass-sweep evaluations, 20 weight-sweep evaluations, and
4 evaluations of the two K=2 models. The original 26 checkpoints and their
matched-K results are reused when the workspace is retained.

The original settings remain in `configs/retrieval_studies_26.yaml`.
The new combined experiment adds seven frozen linear segmentation evaluations
for **79 total**. See [the complete setup and protocol](ablation79.md).

Standalone retrieval commands remain supported:

```bash
python -m mpsae.benchmark.retrieval_studies --plan
python -m mpsae.benchmark.retrieval_studies \
  --workspace /workspace/retrieval-studies \
  --source-root /workspace/retrieval-studies/source --download-embeddings
python -m mpsae.benchmark.retrieval_studies \
  --workspace /workspace/retrieval-studies --export-only
```

`--studies mass`, `--studies optimizer`, and `--studies extreme` select stages.
The standalone `scripts/retrieval_server_onefile.py` still installs only the
retrieval dependencies and runs these 72 evaluations. To run the additional
segmentation probe, use `scripts/setup_ablation79.sh` and
`scripts/run_ablation79.sh` instead. Results are local; no artifact upload.

Inference grids are explicitly keyed by study/backbone/training K in YAML.
Changing inference does not change a real training identity. Keep both K
columns in any exported analysis; the reporter groups them separately.
