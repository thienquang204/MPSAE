# Scientific audit: MRL, CSR, and MPSAE on ImageNet-1K

## Status of the comparison

The earlier experiment did not support a defensible claim about MPSAE versus
the published MRL and CSR results. It mixed torchvision ImageNet weights,
short local fine-tuning, a single CSR training support, and altered CSR loss
details. Those choices define a useful ablation, but not the FF2048 experiment
reported by Wen et al.

The corrected primary protocol is now deliberately narrow:

```text
released FF2048 ResNet-50 embeddings
  + published MRL Table 4 reference
  + independently trained fixed-K CSR reproduction
  + independently trained fixed-K MPSAE comparison
  -> unit-normalized exact L2 1-NN on ImageNet-1K
```

This separation matters more than choosing a nominally stronger torchvision
checkpoint. The scientifically relevant backbone is the exact FF2048 feature
source used by the paper, not whichever ResNet-50 weights have the highest
torchvision classification accuracy.

## Evidence hierarchy

The implementation follows sources in this order:

1. The CSR paper's Table 3 for reported image hyperparameters.
2. The authors' released visual source for details omitted by the table.
3. The released FF2048 NPZ files for the actual dense inputs.
4. The original MRL repository for MRL checkpoint provenance and its 40-epoch
   FFCV ResNet-50 recipe.

The CSR repository README currently shows example defaults that differ from
Table 3 (for example, learning rate 4e-4 and weight decay 0). Those examples
are not used for the strict reproduction. Table 3 specifies learning rate
4e-5 and weight decay 1e-4.

## Why the former MRL result was not recoverable

The original MRL result is not a 10-epoch fine-tune of torchvision
`IMAGENET1K_V1`. The released MRL recipe trains ResNet-50 models for 40 epochs
with FFCV, blur-pooling-related model code, distributed training, and a cyclic
learning-rate configuration. Appendix C of the paper reports learning rate
0.475, batch size 256 per GPU on two A100s, momentum 0.9, and weight decay
1e-4. (The current repository README example says 0.425, so the paper and
README are not identical.) Its 10-epoch, lr=0.1 fine-tuning experiment is a
separate Appendix K.1 ablation and cannot replace the main FF2048/MRL
baseline.

The corrected harness therefore does not locally retrain MRL in paper mode.
It records Table 4 as a published reference, with `locally_retrained=false`.
This is more honest than assigning the MRL name to a different training
recipe. A future local MRL reproduction should be a separate project using
the original FFCV pipeline and released checkpoint conventions.

## Why the former CSR result was not recoverable

Several seemingly small changes alter CSR materially:

- The input must be the released FF2048 ResNet-50 embedding, not torchvision
  ResNet-18/50 output.
- The latent width is 8192, not 8196.
- Each K in {8, 16, 32, 64, 128, 256, 512, 1024, 2048} has its own trained
  model. Evaluating one K=32 model at all sparsities is a different study.
- Table 3 uses batch size 4096, 10 epochs, Adam learning rate 4e-5, weight
  decay 1e-4, and epsilon 6.25e-10.
- The reconstruction objective is K plus 4K, weighted 1 and 1/8.
- The auxiliary residual loss has coefficient 1/32 and is normalized by the
  residual target's mean-predictor MSE.
- The dead-latent threshold is 30 Top-K forward-counter updates, matching the
  released model's activity statistic.
- NCL has coefficient 0.1, duplicates the deterministic representation,
  L2-normalizes both views, and uses temperature 0.2.
- Pre-bias starts at zero. It is not initialized to the empirical feature
  mean.
- The released tied decoder is not projected to unit norm after every update.
- The final incomplete minibatch is retained.
- Paper mode uses float32 with AMP and TF32 disabled.

All of these are now represented explicitly in code or strict validation.

## Reference values and integrity gate

Table 4 reports the following exact-L2 1-NN top-1 values:

| K | MRL | CSR |
|---:|---:|---:|
| 8 | 62.19 | 67.78 |
| 16 | 67.91 | 69.17 |
| 32 | 69.46 | 70.15 |
| 64 | 70.17 | 70.94 |
| 128 | 70.52 | 70.99 |
| 256 | 70.62 | 71.31 |
| 512 | 70.82 | 71.29 |
| 1024 | 70.89 | 71.30 |
| 2048 | 70.97 | 71.20 |

The full FF2048 representation is 71.19. The harness evaluates that point
before interpreting learned sparse representations. If it is not recovered
closely, likely causes include the wrong archive, label/order corruption,
normalization differences, incomplete gallery/query splits, or a retrieval
implementation mismatch. Sparse-method conclusions should be withheld until
this gate passes.

## Fair MPSAE comparison

MPSAE is not in either cited baseline paper, so there is no paper-defined
MPSAE hyperparameter set. Fairness is achieved by holding the shared
experimental substrate fixed:

- identical released FF2048 examples and labels;
- identical tied SAE parameterization and initial state;
- identical K and 4K reconstruction terms;
- identical normalized auxiliary loss and dead-latent mechanism;
- identical Adam settings, batch exposure, and 10 epochs;
- one independently trained model per evaluation K; and
- identical encoding, row normalization, and exact 1-NN evaluation.

The 2K representation is used only as the third MMPOT view. It does not add an
extra reconstruction term, which would otherwise give MPSAE a stronger base
objective than CSR. Extra MPSAE epochs are zero in the controlled protocol.

The MMPOT mass, entropy, solver tolerance, microbatch size, and loss weight
remain method-specific experimental parameters. They must be disclosed with
the result. The current `.env.example` retains the locally selected MMPOT
weight of 2.0; that value is not from the CSR or MRL papers. If it was chosen
after observing final ImageNet validation accuracy, the reported comparison
should be described as tuned on validation rather than confirmatory.

## Interpretation of the OT objective

For one embedding, MPSAE forms non-negative Top-K, Top-2K, and Top-4K codes.
After L2 normalization, their three-way circular-variance cost is

```text
C(i,j,k) = (2/9) * [d(z_K_i, z_2K_j)
                  + d(z_K_i, z_4K_k)
                  + d(z_2K_j, z_4K_k)]
```

The regularizer compares the identity partial coupling against an
entropy-regularized three-marginal partial-transport optimum. The optimizer's
transport plan is detached, while gradients flow through the cost according
to the envelope argument. This is a coherent implementation choice, but it
does not by itself guarantee better semantic 1-NN accuracy. The controlled
experiment is needed to distinguish a useful inductive bias from added
optimization complexity.

## Claims that are and are not supported

After an actual run, the following claims are supportable:

- how closely the local CSR implementation reproduces Table 4 at each K;
- whether MPSAE improves over that locally reproduced CSR under matched
  training exposure; and
- how MPSAE compares descriptively with the published MRL curve.

The following claims are not supported by a single run:

- statistical superiority across random seeds;
- a locally reproduced MRL result;
- a ResNet-18 MRL comparison; or
- a causal benefit from MMPOT without a matched loss-component ablation.

The dedicated ResNet-18 study now enforces three independent matched seeds
and reports mean plus sample standard deviation. Hyperparameters should still
be frozen before those confirmatory runs.

## ResNet-18 boundary

The CSR appendix's ResNet-18 experiment is an input-dimension ablation with
timm `resnet18.a1_in1k`, d=512, h=8192, batch size 1024, gamma=1.0, and K in
{8, 16, 64, 256}. It is not the main FF2048 setting and has no matching MRL
curve. The repository keeps torchvision ResNet-18 available only through the
explicit `torchvision_ablation` protocol.

The paper-matched part is implemented separately as
`paper_rn18_study`. It uses the exact timm model identity, the Table 9
hyperparameters, independent fixed-K training, and three seeds for both CSR
and MPSAE. The paper does not publish those seed values, so 1227, 1228, and
1229 are explicitly a predeclared local set rather than attributed to the
authors. The full FF512 result is measured locally. The released FFCV
preprocessing is represented by a semantic torchvision translation, including
its validation crop-at-224/256 then resize-to-256 geometry; the metadata does
not claim bit-exact FFCV. Because Figure 4 does not publish exact numeric
coordinates, it is not used as a numeric pass/fail gate. The released
extractor also randomly drops its final incomplete training batch, but does
not publish the ordering/seed needed to identify the same 143 omitted images.
The local study keeps the complete split for both methods and records that
difference. No MRL result is synthesized.

## Artifact to use in the thesis

The retrieval-time experiment now requires GPU execution for both the dense
FAISS reference and sparse CSR/MPSAE inverted-index search. Sparse retrieval
keeps the complete posting index on the selected GPU and visits postings
only for active query coordinates. Its timed region includes initialization
of the bounded score buffer, product accumulation, top-1 reductions, and
merging gallery chunks. Chunking controls temporary memory; it does not
stream gallery arrays from the CPU. Memory exhaustion is a failure rather
than a fallback that would silently change the benchmark.

Search timing excludes encoding, normalization, transfers, CUDA extension
compilation, index construction, and label lookup for every method. Each
distinct batch shape receives untimed warmups. Timed repetitions are
interleaved by query batch and accumulated across the complete query set;
measurements report medians and spread using both CUDA events and
synchronized wall clocks. Dense and sparse methods default to a
query batch of 32 on the same GPU. Per-query numbers are amortized search
costs at that batch size and must not be described as single-query latency.
The sparse search still initializes and reduces score buffers; performance
depends on posting-list lengths as well as K. No speedup is established by
source-level sparsity alone.
Zero-norm codes are rejected before normalization; valid gallery candidates
with no support overlap remain part of exhaustive search. Atomic float32
accumulation can change near ties across repetitions, and bitwise agreement
with dense FAISS is not claimed.

The published MRL accuracy curve has no local timing. Any speedup claim must
name a locally measured dense reference with the same hardware and timing
scope. Sparse storage includes coordinate indices and posting metadata. The
FF2048 integrity-check timing is reused as the local Dense FF2048 point at
dimension 2048 in timing tables and plots, separately from published MRL.
Actual sparse chunks are additionally bounded by `4*1024**2` compact entries
to keep the builder's sorting-memory estimate near 384 MiB; chunk counts and
sizes are recorded rather than assumed from the requested row limit. The
GPU retrieval implementation has not been executed or compiled as part of
this revision, per the user's instruction to implement by logic only. No
runtime improvement has been verified.

The primary result is
`resnet50/paper_ff2048_reproduction.csv` (and its Markdown companion). It
contains, per K, the published MRL and CSR values, reproduced CSR, CSR
reproduction gap, MPSAE, and MPSAE deltas. The annealed `csrv2` and `mpsaev2`
arms are exploratory and should not replace this fixed-K table in the main
comparison.

The corresponding ResNet-18 artifact is
`resnet18/paper_rn18_study.csv`, aggregated at the suite root as
`rn18_study_per_budget.csv`. It is a separate CSR/MPSAE backbone study and
must not be merged into the FF2048/MRL table.

This audit is based on static source and paper analysis. No training,
evaluation, import, or test command was executed while making these changes.
