# Expanded-source NVIDIA / DigitalOcean AMD comparison

Root admitted both 12-job campaigns and compared every retained raw training
state over all 128 steps. **Bitwise equality and held-out learning passed.**
The independent first-step FP64 gradient/AdamW checks passed on both GPUs.

- NVIDIA: RunPod RTX4090, sm_89, source `eac39c36`, run2.
- AMD: DigitalOcean MI325X VF, gfx942, same source, run6.
- Identical source inventory: 258 files, including 45 transitive Mamba files.
- Model: two blocks, 34,944 FP32 parameters; pinned Tiny Shakespeare bytes,
  fixed initial weights and token order, full forward/backward/AdamW steps.
- Held-out loss: 5.5412986278533936 → 2.8436418771743774 on both devices.
- Compared: all raw parameter/gradient/moment/flag/counter/loss state, input
  schedules, held-out tokens/losses and final checkpoint bytes.
- Final checkpoint SHA256:
  `a9858cd59b424b77b897d0171d5f225d63f3f161fcdee02049bd6c04cfd8a6dc`.
- Both rentals were deleted and verified absent. Remote jobs used two-core
  affinity and memory/time guards; the root file comparison used two-thread
  limits and a 1 GiB sampled RSS/60-second bound.

[Full result](comparison.json), [root receipt](root-compare.guard.json).
Comparison SHA256:
`7178c201023f29639e01dd8e786040e7511edb080b28e0b9299305b2e505cf6d`.
The exact invoked root script, wrapper and comparator modules are retained
beside the result. The recorded invocation used absolute local paths.

This new record is continuous training only. AMD also has a matching head64
capture, but cross-vendor continuation on the expanded-source round has not
run. The earlier bidirectional-resume result remains its own historical proof
with the supplemental transitive-source audit. Metal, larger models and
FAST/IDENTICAL speed comparisons remain separate. No three-vendor training or
performance claim is made here.
