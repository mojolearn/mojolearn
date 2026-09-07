# NVIDIA byte-LM checkpoint campaign

Root execution only, two CPU cores/threads and one GPU job at a time.

## head1: retained allocator failure, not a training result

Source `102e6bf56f1adec0ae9f22a1b6083d3cb0212d28`, exact retained NVIDIA
binding from the admitted continuous campaign. Setup and preflight passed;
`head64` exited 1 during initial held-out evaluation, before training.
The CUDA allocator reported a 4 GiB free block but a 1 GiB maximum cache,
then refused a 136.5 KiB allocation. Guard exit was ordinary child failure,
not memory-limit termination. No training or identity admission.

[Original log](head1/remote/byte-lm-resume/head64.log) and
[controller](head1-controller.log) are retained. Pod `46zarb0z38j30a` was
deleted with HTTP 204 and verified absent with HTTP 404, 57 minutes left.

The next transport snapshot uses a bounded 4 GiB CUDA pool; AMD retains its
successful 1 GiB setting. External memory, CPU and wall-time guards remain.
Any retry is a separate output record and must pass full raw comparison.

## head2: pool override still refused

Source `c9d1165dfbcde0a3931764ed3fe2aa3583f860c4`, 4 GiB CUDA pool override.
Setup/preflight passed, but initial evaluation again refused a 136.5 KiB
allocation: the diagnostic now showed a 16 GiB free block against a 4 GiB
maximum cache. Thus the simple minimum-block interpretation of head1 was
incomplete; the exact allocator cause is not established. No training pass.
Pod `tr7iyol9d2zogd` was deleted 204 and verified absent 404 with 57 minutes left.
Original head2 raw logs and controller are retained.

Next attempt restores the NVIDIA continuous baseline's runtime defaults,
with the external 85% GPU-memory guard unchanged. The successful AMD 1 GiB
override remains AMD-only; no progressively larger CUDA pool is attempted.

## head3: source-download timeout

The public GitHub fetch hit its 300-second timeout; no model ran. Source
`3d255241d84c5ba856a3e5eabab942887d88b9ae`. Pod `c9ocwqjoq3l0hp` was deleted
204 and verified absent 404 with 54 minutes left. The fetched source was not
admitted. Head4 uses the same pinned commit and existing bounded SSH archive
transport, without changing numerical source or the NVIDIA runtime defaults.

## head4: 64-step head passed and root raw comparison passed

Source `3d255241d84c5ba856a3e5eabab942887d88b9ae`, exact retained NVIDIA
binding and restored baseline runtime defaults. Setup, preflight, head64 and
remote verification all passed. Root then authored a compact handoff only
after admitting the original full baseline and comparing every raw head step
against its first 64 steps. This was a bounded file-only check after deletion.

Checkpoint SHA256: `264f189179b5a3b8a3cbbceb0f54209fb657ff7dacf701b4f8a760c693da8ff0`.
[Root author record](../2026-09-06-root-byte-lm-cross-vendor/handoff-nvidia-head64-author.json).
Pod `blw55nh4c54rqy` was deleted 204 and verified absent 404 with 54 minutes left.
Foreign continuation/control remain required for full protocol admission.

## resume1: AMD-to-NVIDIA continuation admitted

Transport snapshot `d5893e2a`, exact retained NVIDIA binding and qualified
runtime defaults. Setup and all five compact jobs passed. Root's
[all-raw reverse comparator](../2026-09-06-root-byte-lm-cross-vendor/amd-to-nv-resume128-comparison.json)
exited 0 with identity and learning admitted, no missing prerequisites, and
an effective missing-moments control. Actual AMD checkpoint bytes were loaded
and continuation matches the NVIDIA continuous run at every resumed step.
Pod `bncobasb8l6l6w` was deleted 204 and verified absent 404 with 51 minutes left.
