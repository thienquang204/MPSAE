# Draft / implementation reconciliation

1. The canonical NYUv2 config currently uses MPSAE lambda=0.5 for SigLIP2.
   The paper's main Table 2 column is lambda=2.0. Paper recipes explicitly set
   2.0 for both encoders and disable the SigLIP2-specific override. The existing
   canonical config is unchanged.
2. The legacy weight-sweep CLI forces matched K. VOC Table 2 has cross-K weight
   results. The paper catalog expands weight values into ordinary patch-scene
   configurations retaining the full inference grid.
3. The draft says at most 400 OT iterations. The implementation permits a
   refinement budget of 8000 beyond its initial 400-iteration solve. Record the
   actual refinement diagnostics before correcting the manuscript. The paper
   planner does not change the numerical solver. Its stopping diagnostics also
   need to be distinguished from the appendix's theoretical L1 residual.
4. VOC has 3 matched and 6 cross-K pairs per encoder, not the parenthetical
   count currently stated in the draft. Across three methods and two encoders:
   54 sparse evaluations + 2 dense references.
5. A dense reference is not an upper bound: several sparse results exceed it.
   Generated reports call it Dense.
6. ImageNet uses DINOv2-S/14 (384-D); segmentation uses DINOv2 Base (768-D).
   These are separate encoders and cannot share feature or SAE caches.
7. The 79-evaluation suite includes seven linear-segmentation evaluations and
   two additional upward-budget K=2-checkpoint evaluations absent from this PDF.
   It remains available; the paper catalog selects only the reported conditions.
8. The historical production workspaces are not in this checkout. Encoder/data
   revisions, actual early-stop epochs, solver refinements and evaluation
   precision must be checked against imported receipts. Current locks describe
   a reproducible recipe snapshot, not proof of historical equivalence.
9. The reference CSV contains explicitly labeled transcription spot checks.
   No value from it may populate an evaluation result. Full table agreement
   requires importing original metrics and reviewing all generated outputs.
10. CSRv2 uses semantic training labels in patch SAE fitting; CSR/MPSAE do not.
    Main ImageNet batch sizes differ by method/backbone. Table 4 matches the
    baseline batch/update schedule but also uses s=1 instead of main s=0.9.
    Preserve these distinctions in captions and interpretation.
