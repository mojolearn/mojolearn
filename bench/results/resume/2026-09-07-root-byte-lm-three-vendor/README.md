# Tiny real-text LM: NVIDIA, AMD and Apple continuous training

**Qualified bounded continuous equality:** all 128 training steps match byte
for byte on RunPod RTX4090 CUDA, DigitalOcean MI325X VF HIP and Apple Metal.
Root compared complete raw parameter, gradient, AdamW moment, flag, counter
and loss state, pinned input schedules, held-out bytes and final checkpoint.
All 258 numerical source files match. The model has two decoder blocks and
34,944 FP32 parameters, trained on pinned Tiny Shakespeare text.

- Held-out loss on all three devices: 5.5412986278533936 → 2.8436418771743774.
- Final checkpoint SHA256: `a9858cd59b424b77b897d0171d5f225d63f3f161fcdee02049bd6c04cfd8a6dc`.
- [Comparison](comparison.json): SHA256 `eb5f8514d76153dc89e62dd6de2eaa3a382cf8afb2d9b7eeb8f5ae4e4e8e6641`.
- Independent first-step gradient/AdamW oracles passed on NVIDIA and AMD;
  no separate Metal FP64 oracle is claimed.

Root alone built and ran Metal serially, using two compiler workers/thread
limits and sampled CPU enforcement. The user explicitly requested removal
of the fixed free-memory threshold. The distinct `macos-root-user-tiny-v1`
policy therefore records zero entry/runtime free-reserve requirements,
while retaining a 2 GiB process-group RSS cap, normal-pressure admission and
monitoring, swap/compression-growth stops, sampled CPU limits, 180-second
build/capture deadlines, watchdog and verified cleanup. Original stricter
policy and failed prelaunch attempts remain unchanged historical evidence.

Build took approximately 20 seconds and the full capture approximately
14 seconds, based on guard sampling intervals; these are operational estimates,
not matched performance benchmarks. All guard exits and cleanup passed.
No swap growth was observed. Darwin does not provide hard CPU affinity or
per-process Metal VRAM accounting; the records state these limits explicitly.

`apple/` retains the native binary, complete raw captures, original command
files, logs, receipts, source hashes, installed package metadata and private
runtime configuration. Generated compiler caches are excluded. Commands retain
the original absolute execution paths. Root comparator and policy sources
are archived beside the comparison.

This result is fixed-shape continuous training, not a large language model,
universal shape certificate, public-wheel qualification or speed claim.
Metal checkpoint resume and separate MLP Metal training remain open.
Earlier NVIDIA/AMD bidirectional resume is a separate historical result.
