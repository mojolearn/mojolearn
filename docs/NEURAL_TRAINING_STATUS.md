# Neural training status — September 7, 2026

## September 7 update: Apple continuous training completed

Root built the Metal tiny byte-LM and ran all128 real-text training steps.
Complete raw states, held-out bytes and final checkpoint match NVIDIA and
DigitalOcean AMD bit for bit. Loss fell from5.5413 to2.8436 on all three.
Evidence: `bench/results/resume/2026-09-07-root-byte-lm-three-vendor/README.md`.
The user explicitly removed the fixed free-memory minimum; the separately
recorded user-tiny policy retained two-thread limits, a2GiB RSS cap, pressure,
swap/compression, CPU and deadline stops, watchdog and verified cleanup.
The earlier blocked/readiness sections below are historical, superseded for
continuous Metal training. Metal resume, separate MLP Metal, and installed
byte-LM wheel coverage remain open. No further paper work was performed.


The latest tiny real-text language-model round passed on **RunPod NVIDIA
RTX4090 and DigitalOcean AMD MI325X** at common source `eac39c36`. Both twelve-job
campaigns and independent first-step FP64 gradient/AdamW oracles passed. Across
all 128 continuous steps, every retained raw state, held-out token/loss and final
checkpoint byte matches. The two-block, 34,944-parameter decoder's held-out loss
fell from 5.5412986 to 2.8436419. The source inventory now contains 258 files,
including 45 transitive Mamba files. Both rentals were deleted 204 and verified
absent 404. [Expanded-source evidence](../bench/results/resume/2026-09-07-root-byte-lm-expanded-comparison/README.md).

**New expanded-source cross-vendor resume and Metal remain unrun.** Actual
checkpoint continuation passed in both directions in the older, separately
qualified NVIDIA/AMD round, including effective missing-moments controls.
[Historical resume records](../bench/results/resume/2026-09-06-root-byte-lm-cross-vendor/README.md)
and [supplemental transitive-source audit](BYTE_LM_TRANSITIVE_PROVENANCE_AUDIT.md)
preserve that scope. This is a small language model, not a large language model
or a general three-vendor training certificate.

A language model is a neural network. Our different architecture is an
**8→16→3 ReLU multilayer perceptron**, which classifies numeric vectors rather
than predicting text. NVIDIA's 16-step synthetic-data experiment passed all
18 integration jobs; loss fell from 1.1047444 to 0.0450696. AMD now passed the
same 18 jobs at frozen `8f6ed41`, and all 16 raw training steps match NVIDIA.
[Saved AMD and cross-vendor evidence](../bench/results/resume/2026-09-07-root-mlp-amd/README.md).
These synthetic labels
demonstrate training mechanics; they do not establish real-data generalization.

## MacBook work and safety

Metal portability source for the byte LM is authored: Darwin build flags,
explicit vendor admission and immutable checkpoint-bytes loading. It is not
yet compiled or GPU-validated. These changes affect the numerical-source
inventory. The expanded common-source NVIDIA/AMD continuous round now passes;
Metal itself still requires its own execution evidence. Old records must not
be retroactively rebound to new source.

The September 7 read-only preflight saw 16 GiB total unified memory, about
137 MiB free/speculative, about 5.3 GiB compressed and no swap in use. System
pressure was normal, but the conservative **4 GiB launch reserve was absent**.
No local compiler or GPU model was launched. A later recheck still had only
about 116 MiB reserve. GPU allocation shares this memory
with applications; GPU use cannot guarantee the laptop will remain responsive.

Root will recheck memory before launch, then use the
[Mac supervisor](MACOS_TINY_JOB_GUARD.md): two-thread limits, one job at a time,
a maximum 180-second job deadline and explicit memory/cleanup gates. Begin
with build/readback and one step, then resume and longer captures only after
those pass. The model is tiny; training compute is unlikely to be the main
effort, but Metal compilation and numerical debugging have no validated time
estimate yet. The separate MLP already has NVIDIA/AMD equality; its Metal leg
remains pending local admission.

Root has since passed 88 host-only portability/safety checks plus seven
subtests, and retained common snapshot `d17c1aa1` for the next numerical round.
Its optional Metal comparator also passed 38 focused checks and accepted both
retained server-GPU resume proofs. These are host-only checks, not Metal training.
[Preparation, tests and memory observations](../bench/results/resume/2026-09-07-root-metal-preparation/README.md).

The earlier two-vendor comparison used RunPod MI300X. The requested
**DigitalOcean MI325X follow-up now also passed all 12 jobs**, the independent
gradient/AdamW oracle, 128 training steps and a matching separate head64 run.
Held-out loss again fell from 5.5412986 to 2.8436419. The rental was deleted
and verified absent. [Saved DigitalOcean result](../bench/results/resume/2026-09-07-root-byte-lm-do-amd/README.md).
Its new source inventory includes all 45 transitive Mamba files (258 files
total); the matching NVIDIA refresh and full raw comparison now pass. The
prepared Metal source agrees with those 258 files but still has no GPU execution evidence. The historical
213-file capture subset and supplemental archive audit are documented in
[the provenance audit](BYTE_LM_TRANSITIVE_PROVENANCE_AUDIT.md).

## Why the NVIDIA installation differs

PyPI **0.6.0 is published** for macOS and AMD Linux. Its Linux wheel contains
HIP native libraries, not CUDA libraries. The earlier NVIDIA packaged candidate
passed 19 of 24 installed jobs, versus AMD's 24 of 24. Later source fixes and
successful NVIDIA training do not insert corrected binaries into an already
published wheel. The new byte-LM native extension also requires a source build
on every vendor today.

NVIDIA wheels are feasible. The next release needs the corrected CUDA payload,
an explicit byte-LM packaging inventory, and installed checks of the actual
final artifact. CUDA and HIP share a Linux wheel platform tag, so the intended
combined wheel must contain and dispatch both payloads.
[Concrete closure plan](NVIDIA_WHEEL_CLOSURE_PLAN.md).

All tests, builds, models, measurements and rentals are root-only. Subagents
author and review source only. New work does not alter published 0.6.0 bytes.
