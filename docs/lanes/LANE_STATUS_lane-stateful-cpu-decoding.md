# lane/stateful-cpu-decoding, 2026-09-16

Evidence: `~/mojolearn-evidence/stateful-cpu-decoding/`.

## What the audit said, and what was actually true

The audit reported that the inference wrappers "REFUSE carried state, `step()`
and cache allocation". That is what the source said and it was NOT stale, but
the reason matters and the audit did not have it: the refusals were deliberate
and honest. `bindings/_mojolearn_neural_host.mojo` exported
`transformer_forward_fresh` and `mamba{1,2,3}_forward_fresh` and nothing else,
which a runtime read-back confirmed:

    LOADED _mojolearn_neural_host ['embedding_forward', 'linear_forward',
      'mamba1_forward_fresh', 'mamba2_forward_fresh', 'mamba3_forward_fresh',
      'mlp_forward_logits', 'neural_host_column', 'neural_host_numeric_mode',
      'neural_host_sabotage', 'neural_host_vendor', 'rms_norm_forward',
      'transformer_forward_fresh']

so a `step()` had nothing to call. One thing the audit missed: since
lane/ship-cpu-host-families every host family ships, so `_mojolearn_mamba_host`
and `_mojolearn_transformer_host` already carried the decode entries and the
GPU-named block classes already decoded on a CPU-only install through
`_backend._HOST_MODULES`. What was missing was the decode on the SHIPPED
inference surface, not on the CPU.

## The central result

One fresh-state forward pass over a sequence against the same sequence decoded
one token at a time with a carried state, compared BITWISE per position:

| model | window | first differing position |
|---|---|---|
| TransformerBlockInference | 0 (full causal) | none, equal at all 16 |
| TransformerBlockInference | 8 (sliding) | none, equal at all 16 |
| Mamba1BlockInference | n/a | none, equal at all 16 |
| Mamba2BlockInference (and dt_limit) | n/a | none, equal at all 16 |
| Mamba3BlockInference | n/a | none, equal at all 16 |
| SambaInference (tied and untied) | n/a | none, equal at all 16 |

There is no divergence to localize, and the reason is a design choice that
predates this lane rather than luck: each block oracle runs prefill and decode
through ONE call site (`mamba_block_oracle`'s own docstring: "the decode step
is this same function at `l == 1` carrying the state -- ONE spelling for both,
which is what makes gate D a theorem"), and Mamba-2 and Mamba-3 keep the open
chunk's rows IN THE STATE and rebuild the working sequence on resume, so the
chunked folds a step runs are the folds the prefill ran. This lane's change
keeps that property by deleting the wrappers' own `forward` rather than adding
a second one.

## The check can fail

`tools/step_vs_full_check.py --sabotage` moves ONE carried cell by ONE ULP and
requires the per-position comparison to fire. All six fire, each naming a
position and two values. It SCANS for the cell rather than assuming one,
because a single ULP in a single cached component is often absorbed:

* the transformer's `k_cache` absorbs it at all of the first thirty-two cells;
  `v_cache[7]` does not, and moves position 6.
* mamba1's `conv_window` absorbs it at cells 0 and 1; cell 2 moves position 7.
* mamba2's `h` is ALL ZEROS inside the first chunk, so a one-ULP move there is
  a denormal that `ftz` flushes back to zero. An arm perturbing it is inert.

A first attempt at this arm perturbed `k_cache[0]` and `h[0]` and reported
BITWISE EQUAL, which is indistinguishable from a pass. That is why the tool
prints the cell's bits before and after and fails the run when no cell fires.

## Columns

`tools/identity_break.py --step-full` adds a `stepfull` part on eight lanes.

* CPU column `cpu-apple-m4`: all eight STABLE
  (`10_lane_verification_base.json`).
* Apple column `arm64` (Metal, through `mac_slot.sh metal`): the five lanes
  run read the SAME hashes, IDENTICAL x2 on all five
  (`16_apple_metal_stepfull.json`, `17_diff_cpu_vs_apple_stepfull.log`). That
  run used the shared checkout's package and Metal bindings with this branch's
  harness, so its `commit` field names this branch while the arithmetic is the
  shared checkout's; nothing on main since this branch's base touches mamba or
  transformer.
* Under `MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1` every stepfull cell reads
  BATCH_MOVED and prints the position and both bit patterns.

Nothing pre-existing moved. Against a CPU column built from this branch's
merge base, every train, infer, model, batch, rlpair and ragged cell of the
nine neural lanes reads IDENTICAL (`13_diff_forkbase_vs_lane.log`).

NVIDIA and AMD confirmation of the `stepfull` part is OWED; no box was rented.

## What incremental decoding costs, measured

B = 1, L = 16, d_model = 32, one core, minimum of three repeats.

| column | transformer | mamba1 | mamba3 |
|---|---|---|---|
| Apple (Metal) prefill, ONE call | 204.72 ms | 140.91 ms | 155.50 ms |
| Apple (Metal) 16 steps | 3611.37 ms (225.71 ms/step) | 3386.71 ms (211.67) | 2464.03 ms (154.00) |
| CPU host route prefill, ONE call | 3.17 ms | 3.18 ms | 25.57 ms |
| CPU host route 16 steps | 5.88 ms (0.37 ms/step) | 7.23 ms (0.45) | 346.35 ms (21.65) |

Decoding sixteen tokens one at a time costs 15.8x to 24x the single prefill
call on Metal, and 1.9x to 13.5x on the CPU host route, for the same bits.
A token costs 225.71 ms on Metal against 0.37 ms on the CPU for the
transformer. This is the shape `a7b3b0393` predicted: the number of host round
trips is what Metal charges for, and incremental decoding multiplies them by
the sequence length. At this size the CPU host route IS the decode path, which
is what this lane shipped.

Mamba-2 and Mamba-3 carry a second, column-independent cost: a step rebuilds
the open chunk, so it is O(chunk) rather than O(1). On the CPU column at
d_model 32, B 2, mamba2 (Q = 256) decodes at 404 ms/step against a 355 ms
whole-sequence prefill, and mamba3 (Q = 64) at 44 ms/step against 52 ms. That
is the price of the resumption design that makes the bits equal, and it is
paid on every column.

## Files

* `bindings/_mojolearn_neural_host.mojo` -- eight new entries.
* `python/mojolearn/neural_inference.py` -- the wrappers stop overriding
  `forward`; `SambaInference` gains `allocate_state`, a stateful `forward` and
  `step`.
* `python/mojolearn/host_surface.py` -- the neural family's export list.
* `tools/step_vs_full_check.py` -- the comparison and its fail-first arm.
* `tools/identity_break.py` -- the `stepfull` part, `--step-full`.
* `python/mojolearn/tests/test_neural_inference.py` -- 22 tests pass, 177 with
  `test_host_surface.py`, the reference comparison included.
