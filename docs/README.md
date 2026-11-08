# Documentation

Working notes for the MPSAE repository. Start with the top-level
[`../README.md`](../README.md) — it states the claim, the ground rules and the
three commands that run everything. These pages are the detail behind it.

| page | read it when |
|---|---|
| [`methods.md`](methods.md) | you need the objectives, the OT matching gap, or what we found in the released baselines |
| [`data.md`](data.md) | you need to know where a pixel or an embedding came from, or what a manifest records |
| [`pipeline.md`](pipeline.md) | you are running, resuming or debugging the pipeline |
| [`outputs.md`](outputs.md) | you are reading a result file, a table or a figure |
| [`server.md`](server.md) | you are setting up a GPU box, or a CUDA/driver combination is misbehaving |
| [`notes/mpsaev2_legacy_A.md`](notes/mpsaev2_legacy_A.md) | you want the MPSAEv2 variant we specified but did not implement |

## The two rules everything follows

**Settings follow the published papers. Algorithms follow the released code.**
Where the released code disagrees with its own paper, the code wins and the
disagreement is written down — [`methods.md`](methods.md) lists eight such
cases. Settings we choose (learning rate, weight decay, batch size) come from
CSR Table 3/9 and CSRv2 Table 11, and are stated rather than assumed.

**Every deviation is recorded, never silently applied.** Each embedding archive
carries a `deviations` list in its manifest, one line per departure from the
released recipe with the reason it was forced. Every archive carries at least
the label-key rename; the timm recipe adds two or three more and the MRL recipe
two. They are all listed in [`data.md`](data.md).

## Status

The rewrite has not executed on a GPU yet. Tests that need torch skip on a
machine without it, and the OT golden fixtures still need a one-off capture
(`capture_golden.py`). The first real `--smoke` run is the actual test.
