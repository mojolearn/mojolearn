# NVIDIA expanded-source byte-LM refresh — September 7, 2026

NVIDIA run2 passed all twelve serial jobs and fetched single-vendor admission
at source `eac39c367beeddb8ba4792551d154673e654ce21`, on RunPod RTX4090 / sm_89.
The expanded capture inventory contains 258 files, including 45 transitive
Mamba files omitted from the old capture subset. Root's independent first-step
FP64 gradient/AdamW oracle and controls passed before the separate 128-step
real-text training run.

The two-block decoder has 34,944 FP32 parameters. Held-out loss fell from
5.5412986278533936 to 2.8436418771743774. The full run and first-step capture
passed the existing admission contract. Relevant evidence:

- [Twelve job statuses](run2/remote/byte-lm-validation/results.tsv).
- [Fetched admission](run2/admission.json).
- [Independent oracle](run2/remote/byte-lm-validation/gradient-oracle.json).
- [Full capture summary](run2/remote/byte-lm-validation/full128/summary.json).
- [Transport and allocator record](run2/remote/leg.txt).
- [Controller](run2-controller.log) and [teardown](run2/teardown.txt).

After both rentals were deleted, root compared all retained raw state across
all 128 continuous steps with DigitalOcean AMD MI325X run6 from the same source.
Parameters, gradients, moments, flags, counters, loss, input schedules, held-out
tokens/losses and final checkpoint bytes agree. Both vendor campaigns passed
independent FP64 checks. See the separate
[expanded-source result](../2026-09-07-root-byte-lm-expanded-comparison/README.md)
and [full comparison](../2026-09-07-root-byte-lm-expanded-comparison/comparison.json).

## Retained failure and safety

Run1's create attempt failed before any model work: its controller recorded
HTTP 200 without a created pod and no matching pod name. Preserve the original
[failed controller record](run1-controller.log); run2's success does not change
that attempt's status.

Run2 retained qualified NVIDIA allocator defaults, explicitly excluding AMD's
1 GiB pool-only override. Locked Pixi installation and subsequent jobs ran under
vendor guards, two-core affinity/two-thread limits, memory bounds and deadlines.
Root alone executed all checks, builds, models and the final file comparison.

Pod `4ae685lmdk95wa` was deleted with HTTP 204 and independently verified absent
with HTTP 404 in the controller. No rental remains from this campaign. This was
RunPod; the matching new AMD campaign used DigitalOcean.

## Scope

This newly admitted result is **continuous training only** on the expanded
source inventory. The AMD head64 capture is separate; cross-vendor continuation
on this new round has not run, and Metal remains unrun. The
[older bidirectional-resume proof](../2026-09-06-root-byte-lm-cross-vendor/README.md)
remains separately qualified, supplemented by the
[transitive-source audit](../../../../docs/BYTE_LM_TRANSITIVE_PROVENANCE_AUDIT.md).
No new three-vendor, arbitrary-model, generation-quality or speed claim follows.
