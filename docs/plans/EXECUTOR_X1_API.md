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
  inference session (record off). It runs on a GPU session only (`Exec.IS_DEVICE`); a host session refuses it.
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
(the unrolled body: the result does not depend on it), recording state 0 after every committed iteration on
the device (`OP_LOOP_APPEND`). Returns `{"tokens": (n, tokens per step) int32, "outputs": final inputs,
"iterations": n}`.

A decode body for a language model: inputs `[ids]` (ids input, `(b, 1)`), nodes embedding -> step ops per
layer (state parameters) -> norm -> head -> `argmax(axis=-1, keepdims=1, form=0)` (int32 ids), output the
argmax. Capture it in an inference session, then `body.decode(first_ids, n=N)`.

From Mojo, a family entry calls `control_prepare[E, KIND](sid, body, 0, 0, sid, sid, n, False, unroll,
iteration, True)` and `control_run`, and reads `ControlResult.history[0]`.

## 5. What X1 did not do (owned by the family lanes)

- No family door moved; no hand forward deleted (plan row X1: "nothing yet, shared layer").
- No Python session method per step op: the family lanes reach them from Mojo.
- Backward through a step op: the family lane records its stage rule (section 2).
