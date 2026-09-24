# The Apple arm for T3 segment 6: this M4 cannot hold the target shape

Evidence only. `mojolearn verify` does not run this and no release depends on
it. Taken 2026-09-24 on the local Apple M4 (Mac16,12, 10 GPU cores, macOS
26.5.2 build 25F84), one Metal job at a time under
`/Users/andrewhendel/CascadeProjects/mojolearn/tools/mac_slot.py metal`,
from the PUBLISHED wheel `mojolearn-0.8.17-py3-none-macosx_11_0_arm64.whl`
(sha256 `b34730762d0646dac6a19a009130b86decca097e250d8e3b409a55d56762029e`,
the run's own `"wheel": "0.8.17"`), whose `identical/_mojolearn_byte_lm.so`
(sha256 `91d48f10cc787958...`) is the Metal build of the same source as the
Linux wheel's CUDA and HIP bindings and carries `set_lr`, `export_raw` and
the device fold.

The question was whether route A's segment 6 (`{"segment": "6", "vendor":
"apple", "steps": 100}` in `~/mojolearn-evidence/gpt3-run/t3_spec.json`)
can run on this Mac's GPU at the recipe's shape: 162,147,840 parameters,
batch 4, length 2048, vocabulary 50,257, K = 64 shards. **It cannot.** The
shape was not shrunk and the one-step comparison against the H100's chain
line 101 was not run; there is no verdict to report, only the memory.

## The numbers

| quantity | bytes | source |
|---|---|---|
| this Mac's physical memory (CPU and GPU share it) | 17,179,869,184 | `logs/metal_device.json` |
| Metal `recommendedMaxWorkingSetSize` | 12,713,115,648 | `logs/metal_device.json` |
| Metal `maxBufferLength` | 9,534,832,640 | `logs/metal_device.json` |
| device memory of a batch 4 x 2048 step at this shape (`tools/lm_ce_alias_probe.py`), measured on an MI325X | 38,661 MB | `bench/results/lm_t1_2026-09-22/amd/README.md` (E7, every sample) |
| this process on the M4 at batch **1 x 128** (the target width, 128 tokens), after one step | 12,746,808,544 | `logs/memory_probe_1x128.json` |
| its lifetime peak (the open copies the host state into the device) | 17,081,887,184 | `logs/memory_probe_1x128.json` |
| of which the host state held before the trainer opened | 5,230,203,080 | `logs/memory_probe_1x128.json` |
| swap in use before / after that one probe | 3.21 GB / 9.37 GB | `sysctl vm.swapusage` |

At 128 tokens, one sixty-fourth of one of the recipe's shards, the model at
this width already peaks at the whole of the Mac's memory and pushed about
6 GB into swap. The trainer's buffers at 128 tokens are 7.5 GB (footprint
after the step less the host state), above half of the Metal working-set
limit before any activation of size. The recipe's shard is 8,192 tokens,
and a step of that shard measured 38,661 MB of device memory on the MI325X, 3.0 times the Metal
working-set limit and 2.25 times the physical memory. The 1.95 GB of state
(parameters, m, v), the gradient, the three rollback shadows and the
embedding copies fit; the activations at batch 4 x 2048 (the [M, V]
cross-entropy buffers alone are 1.65 GB each at M = 8,192) do not. This is
what `docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md` section 7 predicted from the
H100's figure; the measurement above is the M4's own.

## What was not done, and why

- **The one-step comparison at the target shape.** Opening a 38.7 GB trainer
  on a 17.2 GB Mac is not a measurement that fails cleanly: a Metal buffer in
  shared storage is paged, not refused, and this Mac hosts the live T3
  driver and the local dead-men of its rentals. The 128-token probe already
  moved 6 GB to swap. No step at batch 4 x 2048 was attempted.
- **A seconds-per-step figure and a 100-step projection.** The only step
  timed is the 128-token probe's (13.3 s for the first step, 1.28 s for the
  second, `logs/memory_probe_1x128.json`); it is dominated by the update of
  162M parameters and says nothing about 524,288 tokens a step. No
  projection is made from it.
- **The `apple` vendor in `tools/lm_run_driver.py`.** Not built: a local
  runner for a GPU that cannot open the shape would be code that never runs.
  With `vendor: "apple"` in the spec the current driver halts at A/6 before
  renting anything (`lm_segment_leg.py render --arm` accepts only `nvidia`
  and `amd`, so the render raises and the driver logs "the rental could not
  be started"). The plan's fallback for this case is route A's segment 6 on
  NVIDIA (section 7).

## What would run it

A Mac whose Metal working-set limit exceeds about 40 GB (the MI325X figure
plus the host's copy of the state), which on Apple silicon means 64 GB of
unified memory or more; or activation recomputation in the byte LM (plan
section 7, a separate lane). Either keeps the recipe, K and the bits.

## Files

- `tests/metal_memory_probe.py`: the probe (one `ParallelByteLanguageModelTrainer`
  per row at the recipe's width, K = 1, `proc_pid_rusage` footprint).
- `tests/metal_limits.swift`: the Metal device query.
- `logs/memory_probe_1x128.json`, `logs/metal_device.json`: the outputs.

Commands (the scratchpad path written `<scratchpad>`, the venv holding the
0.8.17 wheel and NumPy):

    python3 tools/mac_slot.py --timeout 900 metal -- env -u PYTHONPATH MOJOLEARN_NUMERIC_MODE=identical \
        <scratchpad>/venv/bin/python bench/results/lm_t3_apple_arm_2026-09-24/tests/metal_memory_probe.py \
        --recipe /Users/andrewhendel/mojolearn-evidence/gpt3-run/t3_recipe.json \
        --manifest /Users/andrewhendel/mojolearn-evidence/gpt3-run/fineweb-tokens/manifest.json \
        --rows 1x128 --out <scratchpad>/probe1.json
    swiftc -O -o <scratchpad>/metal_limits bench/results/lm_t3_apple_arm_2026-09-24/tests/metal_limits.swift
    <scratchpad>/metal_limits
