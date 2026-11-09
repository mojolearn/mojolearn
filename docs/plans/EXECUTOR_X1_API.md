# Executor X1 API: state, block step ops, weights as parameters, decode

Lane `x1-executor-state-and-step-ops` (2026-11-08), plan
`docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md` section 2.1 and row X1. This note is for the family lanes
(X2, X4-X9, N1-N6) that branch from X1's head. The signatures below are stable; a lane that needs a change
asks the orchestrator rather than editing them in place.

## 1. Carried state

A state tensor is a session-owned upload whose storage is written in place. State is a property of the
buffer (`OpsSession.buffer_kind`, by address), so a captured program's parameter leaf over the same storage
is state too.

| Mojo (`autograd/ops_api.mojo`, `OpsSession[E]`) | Python (`mojolearn.ops`) |
|---|---|
| `declare_state(h)`: an upload (`upload`, `upload_nd`, `upload_ids`) becomes state; int32-word uploads become integer state | `Session.state(x, ids=False)` |
| `is_state(h) -> Bool` | |
| `load_into(h, src)`: overwrite (int32 words keep their words) | `decode(..., state={h: array})` |
| `upload_block(code, ints, part, srcs, words) -> h` (section 3) | |

Rules: state takes no gradient; `capture` records a state parameter as `PROGRAM_PARAM_STATE` or
`PROGRAM_PARAM_STATE_WORDS` (`autograd/program.mojo`), and a program holding one saves as record version 3
and loads back as state (`ops_load_program`). A write from outside a step op (`load_into`, an optimizer
`step`, `declare_state`, `free`) drops the step ops' resident structs over that buffer
(`BlockStepScratch.forget`).

`Program.while_loop` now carries int32-word state (token ids, positions): each body output keeps its state
input's kind (refused by name otherwise). `iteration=True` with an ids iteration input gives the body an
int32 word (`OP_LOOP_ITER_WORD`), so `take`/`index` can select token t of an uploaded slab.

## 2. Block step ops

`autograd/block_step_ops.mojo`, session codes 81..84 (`PROGRAM_MAX_CODE = 84`, record version 3):

| code | name | tensors (in order) | ints | floats |
|---|---|---|---|---|
| `OPS_M1_STEP` 81 | `mamba1_step` | x, weights, state | `[b, d_model]` | none |
| `OPS_M2_STEP` 82 | `mamba2_step` | x, weights, state, words (1: buf_len) | `[b, d_model]` | `[dt_lo, dt_hi]` |
| `OPS_M3_STEP` 83 | `mamba3_step` | x, weights, state, words (2: buf_len, pending) | `[b, d_model]` | none |
| `OPS_ATTN_STEP` 84 | `attention_step` | x, weights, kv, words (1: cached tokens) | `[b, d_model, n_heads, n_kv, head_dim, intermediate, s_max, window]` (+ the 17 `BlockOptions.to_params` words) | none |

- `x` holds `b * L * d_model` floats (any shape); the result has x's shape. L = 1 is the decode step (the
  resident session's step body), L > 1 a prefill or chunked continuation (the session's forward body). One
  op per block for `forward`, `step`, `decode`.
- Run it like any op: `var y = s.op(OPS_M3_STEP, [x, w, st, words], [b, dm], List[Float32]())`, inside an
  inference session (record off). It runs on a GPU session and (lane x7b-step-ops-host-arm) on a host session, where
  the host arm runs the host block bodies (the CPU identity column's oracles) over the same packed tensors.
- The bodies are the resident sessions' unchanged: same structs, same block forward entry, same launch set
  and order. Bits are the session's.
- `capture` keeps a step op only when an output depends on its result (as every op). Name the block output
  (or the tokens computed from it) as an output; the state update happens because the op runs.
- Resident structs: built once per (weights, state, words) buffer triple at its first run (the block's
  weight refusal runs then), kept in `OpsSession.steps`, dropped on any outside write.
- Training through a step op: the op runs with recording off. A family lane that trains through it records
  the block with its stage rule (`rules_mamba`, `rules_mamba3`, `rules_transformer`) as a `Tape.custom_nd`
  node over the stage buffers; the step op keeps its stages per call (Mamba-2/3, L > 1) or resident (L = 1,
  Mamba-1 and attention), in `BlockStepScratch` entries (`_M1Block.stages`, `_AttnBlock.stages`).
- Training through `OPS_ATTN_STEP` (lane x7-byte-lm-on-executor, 2026-11-09): in a RECORDING replay (Program.fit) the op
  is a zero-state prefill at position 0 with its stages kept in the (weights, state, words) entry, recorded as one custom
  node over [x, weights] with tag `ATTN_STEP_TAG` (108); `StepOpRules` (`autograd/rules_step_ops.mojo`, ImportRules plus
  that tag) runs the transformer stage rule over the kept stages and writes d_x and the packed weights' gradient.
  Program.fit backpropagates with StepOpRules; `grads_out` copies the last minibatch's gradients out. The byte LM trainer
  (`training/byte_lm_program.mojo`) is the first family on it. The Mamba step ops still refuse a recording session.
- The host arm (lane x7b-step-ops-host-arm, 2026-11-09): `OpsSession[HostExec]` runs all four ops through
  `host_block_step_run` (`transformer_block_oracle`, `mamba_block_oracle`, `mamba2_block_oracle`, `mamba3_block_oracle`):
  the weights are read from the weights tensor at every run, the state is read in and written back after a successful run,
  the integers are the words tensor's. A recorded `OPS_ATTN_STEP` on a host session is the same zero-state prefill with
  its stages kept (`host_attn_step_train_run`), and `StepOpRules` runs `transformer_block_backward_oracle` over them
  (`host_attn_step_backward`). Reports of a host Mamba entry: `host_block_step_reports`. A program that trains or decodes
  through the step ops runs unchanged on the CPU column (the byte LM's `ByteLmProgram[E, KIND]` is the first).
- Adding an op: one row in `_info`, one `_run_*` and one arm in `_step_run`; codes stay contiguous
  (`OPS_STEP_FIRST..OPS_STEP_LAST`); raise `PROGRAM_MAX_CODE` and name the op in
  `autograd/program_tools.mojo program_op_name`. No new `elif` in `_op_run_tensors`.

## 3. Weights as program parameters

A block's weights are ONE uploaded session tensor, `(1, n)`, the pieces packed in the block struct's field
order, each at a `block_slot` boundary (256-byte aligned, plus the Mamba poison band), read in place at every
run. The carried float state is packed the same way into one state tensor.

```mojo
from autograd.block_step_ops import (OPS_M3_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS,
    block_weight_sizes, block_state_sizes, block_offsets, block_guard, block_part_floats)
var ints: List[Int] = [b, dm]
var w = s.upload_block(OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS, weight_srcs, List[Int]())  # 9 CastIn, field order
var st = s.upload_block(OPS_M3_STEP, ints, BLOCK_PART_STATE, state_srcs, List[Int]())    # 10 CastIn, declared state
var q = s.upload_block(OPS_M3_STEP, ints, BLOCK_PART_WORDS, List[CastIn](), [q0, pending])
```

- `block_weight_sizes(code, ints)` / `block_state_sizes(code, ints)`: logical piece lengths (Mamba
  `_m{1,2,3}_weight_sizes` / `_state_sizes`; attention: the options constructor's twenty weights, an absent
  optional piece 0 and an empty `CastIn`; state `[keys, values]`, each `b * n_kv * capacity * head_dim`).
- `block_offsets(sizes, block_guard(code))`: piece starts and the total. A family that reads a piece back
  (`state_dict`, export) downloads `[offset, offset + size)` of the tensor.
- The family's resident-session copy is what the lane deletes: the program's parameters ARE the resident
  copy. `parameters()` / `state_dict()` download on request.

## 4. Decode

`Program.decode(*arrays, n, state=None, unroll=1, iteration=False)` (`python/mojolearn/ops.py`, thin):
the native door `ops_program_decode_py` (`providers/python/autograd_program_control_pyapi.mojo`) loads
`state` into the session's state tensors, then runs the body `n` times as the bounded loop with no condition
(`control_prepare(..., condition_id=0, history=True)`), `unroll` iterations enqueued between two status reads
(the unrolled body: the result does not depend on it; `None` / 0 is the chunk rule of section 5), recording state 0 after every committed iteration on
the device (`OP_LOOP_APPEND`). Returns `{"tokens": (n, tokens per step) int32, "outputs": final inputs,
"iterations": n}`.

A decode body for a language model: inputs `[ids]` (ids input, `(b, 1)`), nodes embedding -> step ops per
layer (state parameters) -> norm -> head -> `argmax(axis=-1, keepdims=1, form=0)` (int32 ids), output the
argmax. Capture it in an inference session, then `body.decode(first_ids, n=N)`.

From Mojo, a family entry calls `control_prepare[E, KIND](sid, body, 0, 0, sid, sid, n, False, unroll,
iteration, True)` and `control_run`, and reads `ControlResult.history[0]`.

## 5. The gate: how a body's writes stop at the loop's stop (lane x1b)

Lane `x1b-executor-gated-state-chunked-loop` (2026-11-08). Every write an iteration makes is behind the loop's
gate, so a loop may enqueue k iterations between two status reads and the result is still the per-iteration
loop's. A body never needs the gate itself:

| writer | how it is gated | where |
|---|---|---|
| loop-carried state (the body's outputs) | `OP_LOOP_COMMIT`, a bit copy when `LOOP_GATE` is open | `_commit_state` |
| the token record, the iteration register | `OP_LOOP_APPEND`, `OP_LOOP_ADVANCE` on `LOOP_GATE` | `control_run` |
| declared state a body op writes in place (a block step op's state, kv and words: any body parameter that `is_state`) | `OP_LOOP_SHADOW` after every body: gate open, the shadow takes the new value; closed, the live tensor gets the committed value back | `control_run`, `ControlPlan.shadow` |
| a block step op's host integers (positions, chunk fill) | after a loop whose bodies ran past the stop, `OpsSession.steps.forget` on every shadowed buffer: the next run rebuilds the entry from the restored words | `control_run` |
| Program.fit's parameters and optimizer moments | `OP_LOOP_SHADOW` once per epoch; the host step counts, bias-correction products and schedule epoch go back to the stopping epoch's (`_FitMark`) | `train_program_control` |
| a step op of the device loop (`LoopStep`, N7's Cholesky / eigh / sort / compaction) | the op takes the gate (`registers.words`, word 0 the done word; core/step_gate.mojo) | `program_loop_device.mojo` |

So a family that writes a body with block step ops or other in-place state writes nothing for the gate: the
executor shadows every declared state parameter of the body when iterations can run past the stop (a condition
and a chunk above 1), and only then. A body enqueued between two reads that raises (a step op's host position
past its cache, say) is judged at a read: if the loop had already stopped, the error is dropped (the
per-iteration loop would never have run that body) and its in-place writes are undone by the closed gate.

The chunk k (`check_every`, decode's `unroll`) is a program value (`ControlPlan.check_every`). `0` (Python
`None`, the default) is the cost rule `loop_auto_chunk` (`autograd/program_loop.mojo`): no condition reads
once at the bound; a body whose declared in-place state exceeds `LOOP_DRAIN_FLOATS` (2**20 floats: its copy
costs more than a drain) reads every iteration and shadows nothing; otherwise `LOOP_AUTO_CHUNK` = 8. The
first iteration is read first, and the loop keeps k only when that iteration kept every state shape. k never
enqueues a body past `max_iterations`; `iterations` is the device's iteration register.

Program.fit (`autograd/program_runtime.mojo train_program_control`) is the same loop with the epoch as the
iteration: `OP_LOOP_OPEN` opens the gate, the row order (`permutation` / iota), the minibatch gathers and the
optimizer steps are enqueued on the device, the loss curve is a gated commit, and `OP_LOOP_FIT_TICK` counts
the committed epoch and applies scikit-learn SGD's stopping rule when `tol` is given (`n_iter_no_change`
stalled epochs). The host reads once per `fit_epochs_per_read` epochs (about `FIT_STEPS_PER_READ` = 16
optimizer steps), and never without a criterion. Returns the epochs run (`n_iter_`). This is the shape the SGD
family moves onto (lane N3b): its losses as loss kinds, `tol` / `n_iter_no_change` as Program.fit's.

## 6. What X1 did not do (owned by the family lanes)

- No family door moved; no hand forward deleted (plan row X1: "nothing yet, shared layer").
- No Python session method per step op: the family lanes reach them from Mojo.
- Backward through a step op: the family lane records its stage rule (section 2).
