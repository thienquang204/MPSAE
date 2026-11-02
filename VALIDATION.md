# Validation status for MMRSAE and the repair research profile

The November 2, 2026 selected-comparison update limits the benchmark and
reports to CSR, CSRv2, DLMMSAE and DLMMSAEv2. The 131-check CPU suite passed
128 checks on its first run; the three new Bash launcher checks initially
failed because the Windows test environment lacked MSYS coreutils on PATH.
After correcting the test environment, all six selected-benchmark checks
passed, covering those three launcher cases as well as canonical naming,
restricted CLI/report method lists, and step-for-step agreement with the
original CSRv2 formula on both backbone dimensions. Thus all 131 distinct
checks passed across the suite and the targeted rerun. Report regression tests
also verify that old `dlmmsae-annealed` results are read as `dlmmsaev2`, the
other ablations are excluded, and source result files remain unchanged.
Launcher checks use `--plan` only; no GPU job or data download was started.
Historical ablation implementations remain in the model library. The existing
source-identity guard still prevents silently resuming a different code recipe.

The November 2, 2026 DLMMSAE and overview-plot update passed **124 CPU tests**
under `tests/` in the existing `machinelearning` environment. The 30 added tests
cover direct context KL, frozen/disjoint reference banks and self-ID exclusion,
adapter gradients independent of residual blending, clean-teacher/noisy-student
RNG, context KL together with sparse KL and angular OT, actual training-K
annealing, exact Adam continuation, training-only data splits and synchronized
training-time accounting. Linear-probe tests cover real CPU classifier learning,
query-label isolation, sparse coordinate preservation, deterministic epoch
checkpoint continuation, and interrupted per-K/full-parent aggregation. Plot
tests generate PNG/PDFs, keep classification distinct from 1-NN, distinguish K
from measured nonzero counts, reject unmeasured latency, and average only
matched backbone sets at the same K. The final complete suite passed after
correcting a test expectation: full dense width is not a measured nonzero count.

Changed Python modules compile, the launcher passes Bash syntax checking, and
`git diff --check` passes. File-writing tests and the MSYS shell check needed
permitted execution outside the Windows sandbox. Nine-method layout fixtures
were rendered and visually inspected under ignored `runs/.plot_layout_preview`;
they are synthetic test values, not experiment results. `torch.cuda.is_available()`
is false in this environment. No ImageNet fit, CUDA retrieval, GPU linear-probe
fit, Docker build, download or dependency installation was run.

The current recipe is `mmrsae-imagenet-context-distillation-v4` with fixed
training references and a separate context RNG in DLMMSAE checkpoints. The
auto evaluation grid now starts at K=2,4. Old protocol/source identities cannot
resume under this changed recipe. End-to-end CUDA correctness, actual accuracy,
compute cost and method improvements remain unmeasured.

The earlier November 2, 2026 landmark-first architecture update passed 93 tests
under `tests/`, all 16 `check_reference` mathematical checks, and one subsequently
added optimizer-resume regression (110 CPU checks in total). Eleven tests in
`tests/test_mmrsae_dictionary_first.py` cover landmark synthesis before the sole
SAE output gate, original-feature reconstruction targets, frozen dictionary
buffers and frontend gradients, clean budget nesting, dense activation masking,
checkpoint restoration/old-architecture rejection, actual training-K annealing,
and KL-bank refresh at budget changes. The real eight-update CPU resume test
restores dictionary/model state, Adam moments, sparse bank keys and mask RNG;
its continuation matches uninterrupted training bitwise across further K changes.
The multi-K tests now initialize the input dictionary and check the K/2K/4K
schedule rather than the superseded auxiliary-B schedule.

The full suite required a permitted rerun outside the Windows sandbox after
temporary-directory access failed with WinError 5, including inside the
workspace. The Bash syntax check likewise passed after a permitted rerun;
the task's temporary workspace directory was then removed. Changed Python
files compile, and whitespace checks pass. CPU accounting probes verify model
buffers and mocked GPU preflight estimates; they do not measure actual GPU use.
The Docker COPY list includes the new `mmrsae_dictionary.py` module. No ImageNet
fit, Docker build, dependency installation, download or CUDA benchmark was run.
That revision used `mmrsae-imagenet-landmark-first-v3`; its checkpoints carry
the fixed training dictionary and reject the former residual-repair architecture.

The November 2, 2026 angular-OT/multi-K update passed all 83 tests under
`tests/` and all 16 `check_reference` mathematical checks in the existing
`machinelearning` Python environment. The 12 new tests cover angular-only model
objectives and rejection of mixed-cost overrides, finite CPU backward, CSRv2's
cosine budget and endpoint loss/gradient agreement with CSR, repaired inference
at multiple K, budget validation, plan/preflight routing, per-K evaluation
dispatch, interrupted-result persistence, legacy report compatibility and PNG/PDF
generation with all seven methods. Retrieval in these new tests is mocked;
synthetic report values are test fixtures, not ImageNet results. The initial
sandbox run passed the computation checks but could not write temporary files;
the permitted rerun passed, followed by the full CPU regression suite.
The MSYS Bash syntax check for `run_mmrsae_imagenet.sh` passed after a permitted
rerun outside the sandbox, which had blocked its signal pipe. Python compilation
and whitespace checks also passed. No dependency was installed, no data was
downloaded, and no ImageNet fitting, Docker build, extension compilation or CUDA
retrieval benchmark was run for this update. The new protocols reject old recipe
identities; these checks do not validate a server's CUDA runtime or measured accuracy.

The November 1, 2026 prepared-embedding update passed 31 CPU data tests in the
existing `machinelearning` environment: 15 new archive/source/routing checks
and 16 existing split/cache/planner/bootstrap checks. They cover direct
ZIP64 NPZ mapping, Fortran and byte-order preservation, streamed compressed
maps, unchanged-file hash reuse and changed-file rejection, local/offline
loading for both backbones, targeted Hub file requests, JSON/LFS hash
agreement, dev/official boundaries, legacy FF2048 overrides, and plan/preflight
without data work. The synthetic Hub client is mocked: no large archive is
downloaded by these tests. Test fixtures use short temporary directories to
avoid Windows' path-length limit; the initial sandbox attempt could not write
those temporary fixtures, and the successful rerun had the required access.
An authenticated read of the real Hub repository metadata confirmed a private
model repository at `db9dd8570abe1db97cdf67c9d7c9852cce6ac902`, with four NPZ/JSON
pairs and archive LFS hashes matching the local receipts. No dependency was
installed, and no Docker image, training job or CUDA benchmark was run.
The proposed broader CPU suite was not run because its execution request
was declined; no new claim is made about those additional checks.
Python compilation checks passed for the six changed/new Python files, and
`bash -n` passed for the MMRSAE, Docker, container and convenience launchers.
The initial sandbox Bash attempt could not create its signal pipe; the
successful syntax-only rerun had the required access. `docker compose config
--format json`, using the public example settings and a dummy W&B key, also
passed and confirmed the prepared-embedding source, model repository, two
persistent mounts and support for an empty Hugging Face token with local data.

The October 31, 2026 neighborhood-cost correction used source review and small
CPU checks with the existing PyTorch 2.7.1+cpu environment. The previous cost
produced negative values down to approximately -3.6e-7 for identical normalized
1,024-landmark profiles; the corrected normalized, bounded affinity returned
no negative entries on that reproduction. Mixed MMRSAE presets now use rho=0.2.
The 15 neighborhood-cost tests and 11 transport-safety tests passed, including
float64 reference/gradient checks, extreme/masked profiles, invalid inputs,
bounded negative-gap refinement, and scalar/batched solver agreement. All 16
existing mathematical checks passed. The existing 30 data/runner/runtime tests
also passed; one cache test initially hit Windows' path-length limit and was
rerun successfully in a short temporary directory. A small mixed-cost CPU
forward/backward and solve at rho=0.2 also passed. No installation, dataset
download, ImageNet fitting, or CUDA test was performed. These checks do not
establish convergence for every entropy/conditioning setting or server build.
The source-only records below describe earlier updates, not this correction.

The Docker Compose addition was checked with Docker Compose v2.38.1 using
`docker compose config --format json`. The check supplied the Compose manifest
through standard input, substituting only its private `.env` file path with
the public `.env.example` and using dummy credential values. It verified
configuration parsing, online W&B, unchanged seeds and development workload,
ImageNet-only routing, selected GPU, persistent bind mounts, custom paths with
spaces, and actionable errors for missing credentials. No real credentials
were read, and no image build, container launch, download, CUDA check or
training run was performed. These checks validate Compose configuration,
not GPU/runtime compatibility or experiment results.

The server convenience update also used source inspection and editing only.
It adds `run_imagenet.sh` initialization, planning, execution and log commands;
optional host data-directory mounting and GPU selection; Python executable
discovery; and LF rules for Linux shell files. The documented server,
Docker, local execution and resume commands have not been run for this update.
No test, installation, download, compilation, training or benchmark was started.

The October 29, 2026 follow-up also used source inspection and editing only.
It repaired environment/stage routing, added config-derived split preparation
and a shell `pilot` sequence, moved GPU preflight ahead of data materialization,
required authenticated local FF2048 inputs, and rejected singleton OT minibatch
settings at planning time. Documentation now distinguishes manual MMRSAE fits
from the controlled `full_theory_detail.md` repair study. None of these changes
has been executed or tested; the commands in the docs are usage instructions.

The MMRSAE naming, ImageNet-only manual runner, data adapter, GPU preflight,
W&B integration, launchers and documentation were implemented with **source
inspection and editing only**. No Python/Bash program, test, syntax check,
installation, download, extension compilation, training or retrieval benchmark
was executed for this update. Source-level reasoning does not establish
runtime correctness, CUDA availability, successful resume or measured speed.
The launcher contains checks that will execute before data/training work when
the user runs it; their presence is not a report that those checks passed.

The historical check record below concerns older repair-profile drafts only.

The final implementation has **not been executed**. The user requested that
remaining work be limited to source changes and logical review, with no further
tests, program runs, shell checks, installation, compilation, or experiments.

Before that instruction, the implementation work included a whitespace diff
check and Bash syntax checks for the three research/Docker launchers. These
checks covered those files as they existed at that time. They do not validate
later edits or Python, CUDA, training, checkpoint recovery, or retrieval.

The earlier standard-library planner and five data-tool checks that did not
create files also passed. Five file-writing checks were blocked by the sandbox;
they produced no conclusion about the data tools' behavior. Those checks were
not rerun after the source-only instruction.

An earlier draft of the mathematical checks reported 13 passes and two failures
in tight solver-accuracy checks. Subsequent source changes address arithmetic
ordering and use double precision for tiny reference calculations. Those changes
and the final test suite have not been rerun; this document does not label the
failures as experimentally resolved.

The staged runner, compact encoding cache, CUDA benchmark adapter, checkpoint
branching, holdout lock, and final source tree remain unexecuted. No ImageNet
archive was downloaded, no training run was started for this implementation,
and no new accuracy, speed, memory, or convergence result is available.

The supplied specifications remain in [full_experiment_detail.md](full_experiment_detail.md)
and [full_theory_detail.md](full_theory_detail.md). Implementation documentation
describes intended behavior visible in source. A later validation session should
first run the mathematical/data checks, then a tiny real training-slice check,
then CUDA backend and resume checks before committing the pilot training budget.
Those are future steps, not completed validation.
