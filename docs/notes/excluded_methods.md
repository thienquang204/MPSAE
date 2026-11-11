# Archived MPSAEv2 and CSR multi-TopK definitions

As of 2026-11-11, these methods are excluded from every experiment and report.
Only CSR, CSRv2 and MPSAE are selectable. Their old implementations and saved
checkpoints remain for historical integrity; this note does not enable them.

The following descriptions record the setup immediately before exclusion.
References to "current", report outputs or training below describe that older
setup, not the active experiment configuration.

### mpsaev2 — fixed-budget OT plus instance contrastive

The current MPSAEv2 keeps the main budget at K for every training step and adds
CSR's instance contrastive loss to the MPSAE objective:

```
L_mpsaev2 = L(K) + L(4K)/8 + NMSE_aux/32
            + 2·gap(z_K, z_2K, z_4K) + γ·C_instance(z_K)
```

The instance contrastive term uses the normalized K code and its identical
duplicate at temperature 0.2, with the backbone's existing γ above. It does not
read class labels. OT retains mass 0.9, entropy coefficient 0.2 and independent
groups of 32; the contrastive term compares the full 1024-row training batch.
There is no additional 2K reconstruction loss. The K code is reused for the
contrastive term, leaving the three activity-counter updates at K, 4K and 2K.

Against MPSAE, the added instance contrastive term is the objective change:
architecture, fixed K, batch size and reconstruction terms match. MPSAEv2
now receives 15 epochs versus MPSAE's 10, so the measured comparison includes
this training-duration difference as well. Against
CSR-multitopK, both the OT term and the presence of `L(2K)/8` differ, and batch
sizes differ on the 4096-row baseline presets. The hybrid is an ablation to
measure whether retaining full-batch discrimination helps; no performance gain
is established by choosing this objective.

MPSAEv2 is displayed as `MPSAEv2`; its internal key and objective variant
remain `mpsaev2` and `fixed_ot_cl_v1`. An otherwise identical 10-epoch
checkpoint continues through epoch 15 with its weights, Adam state, RNG
and activity counters preserved. Only the stopping point changes; the
constant learning rate and fixed-K loss are unchanged. A different loss,
optimizer, seed or dataset still prevents continuation. The completed
15-epoch fit receives a new signature, so E1/E2 and downstream results
are recomputed for the final weights. Interrupted extensions resume.

This replaces the formerly implemented **variant B**, whose main budget
annealed and whose regularizer was OT only. Old variant-B checkpoints cannot
resume as the current MPSAEv2 because their objectives differ. Only CSRv2 uses
annealing in the current method set. Variant A (fixed main budget, annealed OT
views) and the retired variant B remain documented in
[`mpsaev2_legacy_A.md`](mpsaev2_legacy_A.md).

The new objective records `variant: fixed_ot_cl_v1` and writes fits under
`runs/<backbone>/mpsaev2__fixed_ot_cl_v1/k<K>/seed<SEED>/`. The logical method
name remains `mpsaev2`. Old unversioned directories are preserved and excluded
from current-method reports; they are neither resumed nor relabeled as hybrid
results. Campaign plans record the method-variant mapping, which separates
new reports from those produced for the retired objective.

### csr_multitopk — the control

MPSAE differs from CSR in two ways at once: it looks at a 2K view, and it
couples the views with OT. Without a control, a gain could be explained by
either. `csr_multitopk` keeps CSR's contrastive term and K/4K reconstruction,
adds `L(2K)/8`, and replaces the third duplicate-K activity update with a 2K
update, so:

* **MPSAE > csr_multitopk** compares OT coupling with supervising a 2K view,
  including batch-size and update-count differences where applicable;
* **csr_multitopk > csr** measures the extra 2K reconstruction and its activity
  update under the same backbone batch preset;
* **MPSAEv2 versus MPSAE** compares instance contrastive plus 15 epochs
  against MPSAE at 10 epochs, with the same fixed K, OT, reconstruction and
  batch settings. It is no longer an equal-duration loss-only ablation.

The report explicitly includes this last comparison: ImageNet writes
`paired_mpsaev2_vs_mpsae`, and downstream `paired` tables include rows with
`treatment: mpsaev2` and `control: mpsae`. Differences are computed within
matching seeds and evaluation scopes; existing baseline comparisons remain.

CSR-multitopK follows CSRv2 App. G's "Diverse Multi-TopK" ablation in spirit, which found
extra reconstruction terms help a little but cost more than annealing.


### MPSAEv2 training duration

MPSAEv2 uses 15 epochs (other methods keep 10). A compatible 10-epoch
fit is extended in the same directory with model, optimizer and RNG state
retained. During extension, `summary.json` is marked incomplete. At the
end, `checkpoint.pt` and `summary.json` agree on the new signature,
`schedule.epochs: 15`, and `next_step == steps == schedule.total_steps`.
For an uncapped run this is `15 * ceil(train_rows / 1024)`. Checkpoints
are still saved periodically; only the final checkpoint is a completed
15-epoch result. Synthetic smoke tests may remain capped; real campaign
checkpoint imports require all 15 epochs. Evaluation caches use the new
signature/checkpoint hash. Existing archived settings remain historical.
