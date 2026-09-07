# AMD MLP learning and NVIDIA/AMD raw training equality

Root ran one RunPod MI300X rental from frozen source
`8f6ed4112a3c140326011f908e8caf14c0dee4af`, using the ROCm 6.4.1 PyTorch
image, its explicit Python interpreter and a 1 GiB runtime pool. Every job
ran serially with two-core affinity/two-thread limits. All 18 jobs and fetched
admission passed. [Job statuses](run1/remote/training-validation/results.tsv).

The separate neural network is a fixed **8→16→3 ReLU MLP, 195 parameters**,
trained on 48 synthetic numeric examples for 16 FP32 AdamW steps. Loss fell
from **1.1047444343566895 to 0.04506960138678551**. Independent reference and
edge checks passed, as did AMD's real-file step-8 checkpoint continuation and
effective missing-moments control.

After fetching and deleting the pod, root ran the frozen raw comparator against
NVIDIA run 4. [Comparison](cuda-hip-comparison.json) reports **identity PASS,
learning PASS**, CUDA versus HIP, with 21,024 compared scalar cells across all
16 steps: loss, logits, input/parameter gradients, weights, moments, flags and
counters. Full source/input/configuration witnesses match. This is a second
architecture beyond the [qualified real-text byte LM](../2026-09-06-root-byte-lm-cross-vendor/README.md).

The reader used NumPy 2.5.2 to inspect stored arrays; both model captures used
NumPy 1.26.4. No model/reference math was recomputed locally. The root file
comparison had a 60-second deadline, 1 GiB sampled RSS limit, two-thread
environment and peak observed RSS of 35,968 KiB.
[Root receipt](cross-vendor-compare.guard.json), [log](cross-vendor-compare.log).

This proves only these two fixed continuous training trajectories. The MLP's
checkpoint continuation here is **same-device** on each vendor; foreign-vendor
MLP resume and Metal remain open. Synthetic-label learning is not a
real-dataset generalization result. No speed comparison was performed.

## Provenance and teardown

- [Launch and frozen source](launch.json); [transport overlay](transport.sh).
  Only orchestration changed: explicit ROCm interpreter and GPU architecture
  forwarding, with the current guarded transport. Numerical sources stayed frozen.
- [Dry-run log](dry-run.log): 50 checks passed before rental.
- [Controller log](run1-controller.log): startup took several minutes before
  SSH became available; no model ran during startup. All original statuses retained.
- Pod `te8216hat3pxi4`: deletion returned HTTP 204; independent readback returned
  HTTP 404, with 55 minutes remaining on the armed 60-minute lease. No rental
  remains from this campaign. This was **RunPod**, not DigitalOcean.

Subagents only prepared source and instructions. Root alone ran all checks,
builds, models, comparisons, provisioning and teardown.

## Paper record

The sibling paper repository now includes `results/mlp-two-vendor-2026-09-07.json`,
generated numbers, one main-result sentence and appendix provenance. Root
rebuilt named and anonymous PDFs, with two-thread limits and a 60-second/
1 GiB sampled RSS bound per build. Both passed, and named output refreshed
the tracked preprint. An initial font-style gate refusal was fixed by using
ordinary text for the source revision; its failed log remains retained.
The new text preserves the historical 180/209 counts and excludes Metal,
foreign-vendor MLP resume, real-data generalization and speed claims.
