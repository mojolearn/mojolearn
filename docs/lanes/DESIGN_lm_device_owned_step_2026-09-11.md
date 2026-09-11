# Device-owned LM training step: design (DEVIATION 2514)

Design pass, September 11, 2026. IDENTICAL only. Nothing in this design
changes arithmetic, fold order, rounding, tie rules or the bytes any step
produces; it changes WHERE the state lives between steps and WHEN each array
is validated. No test, build or benchmark ran on the Mac. This file is the
only artifact of the lane; no source was touched.

Parent: [HANDOFF_ai_classical_identical_next_2026-09-10.md](HANDOFF_ai_classical_identical_next_2026-09-10.md)
section 1 ("design a device-owned step API with a lean result and explicit
state, gradient and checkpoint exports"). Measurement it answers:
[BRIEF_lm_step_memory_2026-09-10.md](BRIEF_lm_step_memory_2026-09-10.md),
"Run 2 results: the target step itemized" (H100, main b3d4f3e0). Session as it
exists: [HANDOFF_lm_session_2026-09-10.md](HANDOFF_lm_session_2026-09-10.md).

Sources read (line numbers are the files as read on 2026-09-11; the binding
was being edited by the lifetime lane, DEVIATION 2513, while this was
written, so binding references are by function name with the pre-2513 line
in parentheses): `python/mojolearn/_byte_lm_impl.py`,
`python/mojolearn/language_model.py`, `bindings/_mojolearn_byte_lm.mojo`,
`bindings/hostptr.mojo`, `training/byte_lm.mojo`, `training/checks/loss.mojo`,
`training/checks/loss_oracle.mojo`, `training/checks/optimizer.mojo`,
`training/checks/optimizer_oracle.mojo`, `training/checks/train_loop.mojo`,
`transformer/impl/llama/fused_attention.mojo`,
`transformer/impl/llama/modeling_llama.mojo`,
`python/mojolearn/tests/test_byte_lm_session.py`,
`python/mojolearn/tests/test_byte_lm_surface.py`,
`tools/byte_lm_session_check.py`, `tools/lm_step_memory_probe.py`.

## 0. The one invariant, and the numbers in one place

**Every array that crosses the host/device boundary is validated at the
crossing, and device-resident state is validated on the device by the
existing native checks at the moment it is written.** Today the same 1.95 GB
(param, m, v) crosses the boundary five times per step and is re-validated by
four layers that each assume the previous one lied; the 0.65 GB gradient
crosses twice. Under this design the state crosses at admission (open,
restore, import), at export (explicit call, close) and never per step.

Target shape (B1, L2048, DM768, 12 layers, V50257): n = 162,147,840;
n floats = 648,591,360 B (0.649 GB); 3n = 1.946 GB; 4n = 2.594 GB;
M*V logits = 411,705,344 B (0.412 GB); ids = 2049 int32 = 8,196 B;
n_tensors = 2 + 9*12 = 110 flags = 440 B.

| | per step, today (measured 38.3 s) | per step, this design |
|---|---:|---:|
| bytes crossing the Python/native/device boundary | 14.03 GB (D2H 9.49 GB: 14n mirrors and refusals plus 0.41 GB logits; binding host copies 4.54 GB: 3n read in, 4n published) | about 45 KB (ids 8,196 B host read + 16,384 B H2D; loss 4 B; 10 scan partial vectors of 2,048 B; flags 440 B; step counter) |
| additional host-internal copies of the same bytes | about 21 GB (Python 18n: validate, tobytes twice, NaN-fill, candidate, gradient array and dict; native 14n List conversions inside `download_f32`) | 0 |
| device-to-device | 1.30 GB (unpack 0.65, pack 0.65) | 3.25 GB (unpack, pack, plus the 1.95 GB shadow copy of section 4) |
| device compute | 0.6 s | 0.6 s, plus about 5 ms of scans and the shadow copy |
| expected complete step at the binding boundary | 38.3 s | 0.6 to 0.7 s (gate: under 2.0 s) |

The "14 GB of host traffic" the task names is the first row's today column:
D2H 9.49 GB plus the binding's 4.54 GB of host-to-host copies. Time
arithmetic for the after column: 10 device scans read 9n + M*V floats =
6.25 GB of HBM at roughly 2.5 TB/s effective, about 2.5 ms, plus about 50
context waits inside the scan helper as it exists (about 1 ms); the shadow
copy reads and writes 1.95 GB, about 1.2 ms. Nothing else per step is above
the noise of one `ctx.synchronize()`.

## 1. Ownership

### 1.1 What the resident session owns

`ByteLMSession` (binding) owns the `DeviceContext` and the `ByteTrainer`; the
trainer's `ByteBuffers.param`, `m_state`, `v_state`, `grad` and the flags
`buf_initialized` are the ONLY authoritative copies of the model, the
optimizer state, the last gradient and the flags while the session is open.
The Python `SmallByteLanguageModelTrainer._state` keeps schema, profile,
registry, config, schedule, `completed_steps` and `next_batch_index`; its
`parameters`, `m` and `v` entries are held ONLY while no session is open
(before the first call, after `close()`, after `load_state_dict`) and are
`None` while a session is open. There is no host mirror between steps.

Between steps the device state can be written by exactly three things, and
each validates what it writes:

1. admission (`byte_lm_session_open`): host Lists validated by the existing
   `byte_validate_state` (`training/byte_lm.mojo:124-137`) and then uploaded
   byte for byte (`_upload`, `train_loop.mojo:1025`);
2. the optimizer kernel inside a step: validated afterwards on the device by
   the existing checks moved to the device (section 2.3);
3. rollback after a failed step (section 4): the restored bytes are the
   bytes item 1 or item 2 validated one step earlier, and the rollback
   re-scans them anyway so that a dead context is detected there and not
   one step later.

Nothing else writes them. The forward copies FROM `param` (`_unpack_block`,
the two `_copy_into` at `byte_lm.mojo:541-543`), the backward writes `grad`
and the per-layer `dw_*`, evaluation writes only scratch, exports copy FROM
the buffers, and the Python layer never holds an address of a device buffer
("No borrowed pointer survives a call", binding docstring).

### 1.2 Phase disposition at the target shape (run 2 numbers)

| phase (run 2) | ms | today | under this design |
|---|---:|---|---|
| py_validate_state | 6,775 | every step | ADMISSION only (the existing `_validate_state`, once per open/restore/import) |
| py_candidate_state | 5,924 | every step | EXPORT only (the same `_validate_state` applied to what arrives) |
| py_gradients_dict | 5,501 | every step | `export_gradients(named=True)` on demand |
| opt_refuse_download | 3,122 | every step | PER STEP, on the device: four scans, no download (section 6) |
| bind_resident_admission | 2,256 | every step | GONE: no host mirror to compare; the scalar admission (profile, step, optimizer bits, flags) stays |
| py_input_unchanged | 1,732 | every step | GONE for state (nothing is passed in); the ids copy is owned by `_array` as today |
| py_alloc_outputs | 1,352 | every step | GONE (outputs are loss, step, flags) |
| mirror_download_after | 1,312 | every step | GONE; `validate_after` becomes device scans (section 2.3) |
| mirror_download_before | 1,256 | every step | GONE; `validate_before` is DROPPED on the resident path (section 2.3, with the argument) |
| py_before_bytes | 1,068 | every step | GONE |
| bind_read_inputs | 1,010 | every step | ADMISSION only (3n host read); per step reads ids (8 KB) |
| bind_publish | 879 | every step | EXPORT only; per step publishes 4 B loss and 440 B flags |
| validate_after | 776 | every step | PER STEP on the device (finite p, m, v; v >= 0) |
| bind_validate_outputs | 767 | every step | GONE (duplicate of validate_after); the 4 B loss and the flags are still checked |
| validate_before | 699 | every step | ADMISSION only |
| bind_validate_inputs | 584 | every step | ADMISSION only (3n); per step validates ids (8 KB) |
| capture_copy | 756 | every step | GONE (no `ByteStepCapture` on the resident path) |
| mirror_download_grads | 459 | every step | EXPORT only (`export_gradients`) |
| py_gradients_array | 419 | every step | EXPORT only |
| ce_refuse_download | 264 | every step | PER STEP on the device: one scan of `logits` (section 6) |
| validate_grads | 181 | every step | PER STEP on the device: one scan of `grad` |
| upload_inputs, loss_download, py_tokens | under 100 | every step | PER STEP, unchanged (16 KB up, 4 B down, 8 KB ids) |
| blocks, head, CE, optimizer, embedding, pack/unpack | about 600 | every step | PER STEP, unchanged |

What remains per step: the ids upload, the loss scalar download, the flags
(a 440 B host copy of `buf_initialized`, unchanged by AdamW), the step
counter, six device scans (grad; param, grad, m, v inside the optimizer's
refusal; logits inside the CE refusal) and four more after the update
(finite param, m, v; negative v), and the shadow copy.

## 2. The lean step result and the explicit exports

### 2.1 Public surface

`LanguageModelTrainer(parameters, *, data_schedule, lr, betas, eps,
weight_decay, shape=None, resident=False, step_result='full')`.

- `step_result='full'`: today's `train_step` dict (`loss, step,
  completed_steps, next_batch_index, flat_gradients, gradients`). On a
  resident session it is produced as lean step + `export_gradients()`, so the
  bytes are identical to today's and the state mirror is still gone.
- `step_result='lean'`: resident sessions only; `train_step` returns
  `dict(loss=float, step=int, completed_steps=int, next_batch_index=int,
  flags=Array[int32, n_tensors])`. No gradient norm: nothing computes one
  today (`identical_clip_grad_norm` is the only norm and the profile refuses
  `max_norm != 0`, `byte_lm.mojo:151-155`); adding one would be a new
  reduction and is out of scope.
- `resident=False` with `step_result='lean'` raises `ValueError`: the
  stateless path destroys its context before returning, so there is nothing
  to export from afterwards (section 5).
- Default flip: after the gates of section 7 pass, `step_result` defaults to
  `'lean'` when `resident=True` and stays `'full'` when `resident=False`.

New methods, all under `self._lock`:

- `export_state() -> dict`: the `state_dict()` shape of today (fresh
  `mojolearn.Array` copies of parameters, m, v, flags plus metadata).
  Downloads 3n once. On a resident session `state_dict()` and `parameters_`
  route through it; on a closed or stateless trainer they read the host
  arrays as today.
- `export_gradients(named=True) -> dict(flat_gradients=Array,
  gradients={name: Array})`: the gradient of the LAST completed step, valid
  only until the next `train_step` starts (the device `grad` is scratch for
  the next backward). Refused (`RuntimeError`) if no step has completed since
  open, restore, import or rollback (`grad_step != completed_steps`).
  `named=False` skips the per-tensor dict (the 5.5 s item is the slicing).
- `export_checkpoint(path)`: `export_state()` then the existing JSON/hex
  codec; `save_checkpoint` becomes an alias. The 2 MiB bound is unchanged,
  so at the target shape this refuses as it does today and `export_state()`
  arrays are the checkpoint path.
- `close()`: exports state to the host first (one 3n download, validated as
  in 2.2), then releases the device; a later call re-admits from that host
  state, which is today's observable semantics at a one-time cost of two
  1.95 GB transfers per close/reopen instead of zero.
- `load_state_dict(state)`: unchanged validation (`_validate_state`), then
  release, then the replacement is host state until the next call admits it.
  Shape-changing restore stays allowed (a new session is opened).

### 2.2 Validation at each crossing (nothing weakened, only moved)

| check (today, where) | admission (open, restore, import) | per resident step | export |
|---|---|---|---|
| dtype, shape n_total/n_tensors, C order, owned copy (`_array`, `_byte_lm_impl.py:132-157`) | Python, unchanged | not applicable (nothing crosses) | Python on the arrived arrays, unchanged |
| all_finite(parameters, m, v) (`_array`), v >= 0 and flags in {0,1} (`_validate_state` :273) | Python, unchanged | device scans after the update (2.3) | Python, unchanged |
| config, registry, schema, cursor (`_validate_state` :251-268) | Python, unchanged | scalar admission in the binding stays (profile, step, optimizer bits, flags) | Python, unchanged |
| `byte_validate_state` host scan (binding `_byte_lm_run` after `_read_f32`, pre-2513 :226) | native, on the 3n Lists before upload, unchanged | device (2.3) | native `byte_validate_device_state` before the download |
| `_require_same_bits` mirror admission (pre-2513 :264-266) | not applicable: there is no mirror | GONE, replaced by ownership (1.1) | not applicable |
| input-unchanged (`_byte_lm_impl.py:498`) | not applicable | GONE for state; the ids copy is `_array`'s own | not applicable |
| `all_finite(out_loss)` :501 and the binding's loss bit test | | Python and native, unchanged (4 B) | |
| completed-step equality :494 | | Python and native, unchanged | |
| eval leaves state byte-identical (:509-511, binding eval readback) | | moved to gate G1 (section 7): no write path to param/m/v exists in `_byte_forward_loss` | |

### 2.3 The trainer's validate_before and validate_after

Today (`byte_lm.mojo:498-506` and `:689-697`) both are host scans of freshly
downloaded 3n mirrors through `byte_validate_state`: finite param, finite m,
finite v, `v[i] < 0` refused, lengths, step bound.

**validate_after becomes device scans**, same predicates, same names, same
order, in a new `byte_validate_device_state(ctx, buffers, completed,
config)`: `device_first_nonfinite(param)` then `(m_state)` then `(v_state)`,
each raising the existing message `"byte LM: nonfinite <name> at <i>"`
(`_require_finite`, `:118-121`) with the index the scan returns; then
`device_first_negative(v_state)` raising `"byte LM: negative second moment"`
(the existing message has no index; keep it, append nothing); then the host
length and step checks unchanged. The kernel is `nonfinite_partial_kernel` /
`device_first_nonfinite` (`transformer/impl/llama/fused_attention.mojo:321-383`,
already used by `_refuse_nonfinite_device` in `modeling_llama.mojo:2482` for
exactly this purpose, measured 2026-09-09) plus a sibling
`negative_partial_kernel` / `device_first_negative` with the predicate
`(bits & 0x80000000) != 0 and (bits & 0x7FFFFFFF) != 0`, which equals `v < 0`
for every non-NaN float (NaN is refused by the finite scan first, as today,
since `v < 0` is false for NaN and the finite loop ran first). Both kernels
read the buffer and write only their own partials vector; the reduction is an
integer minimum over indices (exact, order-free); no float is produced and
nothing they write is read by any recorded stage. They are bounds in the
sense of `device_absmax`'s docstring ("A BOUND, not a profile value: nothing
here reaches the card"). This satisfies "must not change any fold that feeds
an output": no output fold is touched.

**validate_before is dropped on the resident path**, with this argument:
at the start of a resident step, `param`, `m_state` and `v_state` hold bytes
that were validated when they were last written (1.1: admission scan before
upload; previous step's device `validate_after`; rollback re-scan), and no
write path exists between steps. A re-scan of unchanged bytes cannot fail.
What it could catch that ownership does not is device memory corruption
between steps, and for that the optimizer's own refusal (`opt_refuse_device_inputs`,
contract 8a, on the device after section 6) scans param, grad, m and v before
any update every step, and `validate_after`'s negativity scan runs after it;
so a non-finite or negative value that appeared in the state by any route is
still refused by name before or immediately after the update, exactly as
today. The stateless path keeps `validate_before` (it validates an upload it
just made; it is the reference and its cost is not the target).

Evaluation on a resident session likewise drops the 3n download and host
validation at `byte_eval_loss` (`:765-768`) by the same argument; the CE
refusal still scans the logits and the loss is still checked finite.

## 3. Mutation rejection

Exports are copies. `byte_lm_session_export_state` copies device to a pinned
staging buffer and then `copy_f32` into the Python-owned `Array` the caller
allocated (the address is validated by `_validate_addresses` as today and
forgotten at return). The session retains no pointer to the exported
memory; the Python layer holds only the metadata. Mutating an exported
array, or the `ids` array after a call, cannot reach the device. The
gradient export is a copy of `grad` in the same way.

Imports are validated before they can reach the device, in this order, and
every failure leaves the live session untouched (`load_state_dict` validates
the replacement BEFORE `_release_session`, `_byte_lm_impl.py:402-405`,
unchanged):

1. Python `_validate_state`: exact key set, schema, profile and registry
   equal to the shape's, config equal to the admitted AdamW form, cursor
   equal to `completed_steps`, dtype and length n_total for parameters, m,
   v, length n_tensors for flags, finite, v >= 0, flags binary. A different
   `model_shape` is a shape-changing restore (allowed: the old session is
   released and a new one opened), a malformed one is refused here.
2. Native `byte_lm_session_open`: `byte_validate_state` on the 3n Lists
   (lengths, finite, v >= 0, step bound), flags exactly 0 or 1,
   `byte_validate_optimizer`, `shape.validate()`, then upload.
3. Native per-step scalar admission (kept from `_byte_lm_run`, pre-2513
   :252-269): profile equality, `completed_steps` equality, optimizer bits
   equality including signed zero, flags equality against `buf_initialized`.
   These are scalars and 110 bools; they stay per step.

There is no path by which a Python caller can hand the session an array
without steps 1 and 2 running on it, because the only native entries that
accept state addresses are `byte_lm_session_open` and the stateless
`byte_lm_run*`, and both validate before any device operation.

## 4. Failed-update recovery and failure atomicity

### 4.1 Failure signals and where each is detected

| signal | detected by | where in the step | state written yet? |
|---|---|---|---|
| bad ids (range, count) | `byte_validate_tokens`, host | before any device work | no |
| non-finite logits (input to CE) | `ce_refuse_device_inputs`, device scan (section 6) | after the head GEMM, before L1 | no |
| non-finite loss | `_require_finite(losses, "loss")` on the 4 B download (`byte_lm.mojo:593`) | after L13 | no |
| non-finite gradient | trainer's grad scan (was `_require_finite(grads)`, `:674`), device | after `pack_grads`, before the optimizer | no |
| optimizer refusal (non-finite param, grad, exp_avg, exp_avg_sq) | `opt_refuse_device_inputs`, device scans (section 6) | first statement of `identical_optimizer_step` | no |
| non-finite or negative state after the update | `byte_validate_device_state` (2.3), device | after `adam_update_kernel` | YES: param, m, v updated in place |
| Python post-check failure (step counter, loss bits, flags) | `_run_impl` | after native success | YES |
| context-level failure (launch error, lost device) | any raise from the driver | anywhere | unknown |

Only the last three need recovery; the first five leave `param`, `m_state`
and `v_state` untouched because every one of them fires before the single
in-place update kernel.

### 4.2 The shadow scheme

True double-buffering (read `(p, m, v)[cur]`, write `(p, m, v)[next]`) is
not available without changing `adam_update_kernel`'s signature: it updates
`param`, `m_state`, `v_state` in place (`optimizer.mojo:1332-1355`), and a
second output pointer would be a new kernel spelling in the optimizer lane's
file. So the design uses a **shadow taken immediately before the update**:

1. `ByteBuffers` gains `shadow_p`, `shadow_m`, `shadow_v` (3n floats,
   1.946 GB at the target) and `flags_before: List[Bool]`.
2. In `_byte_step_device` (the shared step body, section 8), after the
   gradient scan passes and before `identical_optimizer_step`: three
   `_copy_into(shadow_x, x, 0, 0, n)` (the existing D2D kernel,
   `train_loop.mojo:1262`), one `ctx.synchronize()`, `flags_before =
   buf_initialized.copy()`, `shadow_valid = True`, `shadow_step =
   completed_steps`. Timer `step.shadow_copy` (bytes 3n).
3. On success the shadow simply goes stale at the next step's item 2.
4. `byte_rollback(ctx, trainer)`: three `_copy_into(x, shadow_x, ...)`
   (copy, not handle swap: the per-layer weight views of memory candidate
   rank 4, if it lands, stay valid), restore `buf_initialized` from
   `flags_before`, `completed_steps = shadow_step`, `grad_step = -1`, then
   `byte_validate_device_state` on the restored buffers. If that raises,
   the context is not answering and the session is marked lost
   (`usable = False`, error prefixed `"byte LM: session lost"`).
5. The binding's step entry wraps the trainer step: any raise after the
   shadow point calls `byte_rollback` before re-raising; any raise before
   it re-raises directly (state untouched). `healthy` (`byte_lm.mojo:419`)
   is set True again after a successful rollback; it stays False only for a
   lost session. Today's "a failed numerical step poisons this object"
   becomes "a failed step is rolled back on the device; a lost context
   poisons this object".
6. `byte_lm_session_rollback(session)` is exposed to Python for the seventh
   row of 4.1: native success followed by a Python post-check failure. The
   shadow is valid from item 2 of step k until item 2 of step k+1, and no
   Python code runs in between, so the rollback is always available there.
   `_run`'s except path calls it instead of `_release_session()`. If the
   session reports itself lost, the Python layer records
   `self._lost_at = (completed_steps, last_export_step)` and every later
   call raises `RuntimeError("Byte-LM session lost at step k; last exported
   state is step j")`.

Cost at the target: 1.946 GB of device memory (lean device total goes from
9.98 GiB to 11.8 GiB on an 80 GB device; still under the 28.4 GiB full eager
fallback figure) and 3.89 GB of HBM traffic per step, about 1.2 ms at H100
bandwidth. Today's atomicity costs 14 GB of host traffic per step (section
0) and it is not even atomic against the last row of 4.1. The failure path's
own cost (another 1.95 GB D2D and four scans, a few ms) is paid only on
failure.

What is honestly weaker than today: a lost context loses the steps since
the last `export_state()`/`close()`; today a host copy of the last committed
state always existed. This is stated in the docstring, reported by the
error, and is the price of holding no mirror. It is not silent.

## 5. The stateless path

`LanguageModelTrainer(resident=False)` keeps `byte_lm_run` /
`byte_lm_run_configured`, `_byte_lm_run`'s read-validate-run-validate-
publish sequence, `byte_train_step` with its `ByteStepCapture`, the mirrors,
`validate_before`/`validate_after` as host scans, and the Python
`_validate_state` / input-unchanged / candidate checks, all unchanged. It is
the reference for every gate in section 7 and the failure-atomic control
(the Python layer commits a validated candidate or nothing). Its cost at the
target (38 s) is not the target of this design and must not be optimized
in the same change, so that the two arms differ in one thing. The only
edit that touches it is the refactor of `_byte_step_admitted` into the
shared `_byte_step_device` body (section 8, step 4), which the 88-array
resident-versus-stateless comparison and gate G1 cover.

## 6. `ce_refuse_device_inputs` and `opt_refuse_device_inputs`

### 6.1 What they check and why they exist

`ce_refuse_device_inputs` (`training/checks/loss.mojo:1247-1305`, DEVIATION
1495): downloads the full `logits` (M*V) and `targets` (M) to pinned host
buffers, copies both into Lists and calls the oracle's own
`ce_refuse_inputs` (`loss_oracle.mojo:231-275`), which checks in this order:
vocab in [1, CE_MAX_EXACT_COUNT], N in [1, CE_MAX_ROWS], `len(logits) ==
N*V`, label smoothing finite and in [0, 1), `refuse_nonfinite("logits")`
(first NaN or infinity by flat index, message shape of `loss_oracle.mojo:42-56`),
then every target in `[0, vocab)` or equal to `ignore_index`. It exists
because loss_check clause (f) at `ecd1a436` measured that a planted NaN
reached 40 recorded cells ("REFUSED BY NAME before any recorded stage",
contract section 8), and because IDENTITY_PATHS row 39 measured three
vendor-shaped NaN payloads, so a stage hash containing one cannot match
across vendors. Its docstring records the device scan as OWED with the
condition "it must produce the SAME message for the SAME first offending
cell".

`opt_refuse_device_inputs` (`training/checks/optimizer.mojo:1103-1183`,
DEVIATION 1496): downloads `param`, `grad`, `m_state`, `v_state` (4n) to
pinned buffers, copies to Lists, and calls the oracle's `refuse_nonfinite`
(`optimizer_oracle.mojo:38-60`) in the oracle's order: `param`, `grad`,
`exp_avg`, `exp_avg_sq` (AdamW) or `momentum_buffer` (SGD). It exists
because optimizer_check clause (f) measured that with clipping off a NaN
planted in a PARAMETER reached `param.out`, the only device refusal living in
the clip pass that does not run at `max_norm = 0`; and because `m`/`v` are
carried state, so an unrefused non-finite poisons a run rather than a step.
Its docstring records the device scan as OWED under the same "same name,
same first offending index" condition, and names `-D
MOJOLEARN_OPT_TRUST_INPUTS=1` as the deliberate profile downgrade that skips
it. The memory brief's rank 7 is the same item.

### 6.2 The device replacement, same refusal semantics

One shared helper, `core/device_scan.mojo` (new): move `NONFINITE_NONE`,
`nonfinite_partial_kernel`, `device_first_nonfinite` out of
`fused_attention.mojo:321-383` (which re-imports them; training must not
import from the transformer implementation), add `negative_partial_kernel`
/ `device_first_negative` (2.3), and add `device_classify_nonfinite(ctx,
buf, idx) -> Bool` (one 4 B readback of the offending element, `is_nan`),
which is the tail of `_refuse_nonfinite_device` (`modeling_llama.mojo:2482`).
Optionally a `DeviceScanScratch` holding the partials buffer and its pinned
host buffer so the ten scans per step do not allocate; the four
`synchronize()` calls per scan in the helper as it exists are then two.

Message identity is made enforceable rather than hoped for: each oracle
gains a pure string function, `ce_nonfinite_message(name, index, is_nan)`
in `loss_oracle.mojo` and `opt_nonfinite_message(name, index, is_nan)` in
`optimizer_oracle.mojo`, and each oracle's `refuse_nonfinite` is rewritten
to raise through it (a refactor of a string, no numbers involved). The
device paths raise through the same function with the index the scan
returned, so there is one spelling of every refusal message.

`ce_refuse_device_inputs` becomes: the scalar checks and the targets loop
run on the host as today, but from a split of the oracle function into
`ce_refuse_shape(n, n_logits, cfg)` and `ce_refuse_targets(targets, cfg)`
(the oracle's `ce_refuse_inputs` calls shape, then `refuse_nonfinite`, then
targets, so its order is unchanged and there is still no restatement);
targets are downloaded (M int32, 8 KB) as today; `logits` is scanned by
`device_first_nonfinite`, the offending element classified, and
`ce_nonfinite_message("logits", idx, is_nan)` raised. Order: shape, logits,
targets, as the oracle. Timer renamed `step.ce_refuse_scan`, bytes line the
M*V*4 read on device. Cost at the target: 0.41 GB of HBM reads, about 0.2 ms.

`opt_refuse_device_inputs` becomes four `device_first_nonfinite` scans in
the oracle's order with `opt_nonfinite_message(name, idx, is_nan)`, the
`MOJOLEARN_OPT_TRUST_INPUTS` early return unchanged (it still means "no
refusal"), timer renamed `step.opt_refuse_scan`. Cost: 2.6 GB of HBM reads,
about 1 ms. Note that on the resident path `param`, `m`, `v` were validated
one step earlier and `grad` one phase earlier; the four scans are kept
because contract 8a is a clause of the optimizer profile, owned by that
lane, and its cost on the device is not worth a profile argument.

### 6.3 What has no device equivalent and stays, with its cost

- The targets range loop and the `ce_refuse_shape` scalars: host, 8 KB and
  a handful of integers per step. Stays.
- `byte_validate_tokens`: host over 2049 ids per step. Stays.
- Flags in {0, 1} and equality with `buf_initialized`: host, 110 bools.
- The step-counter bound and optimizer-bit equality: host scalars.
- The loss finite test: 4 B download. Stays (it is the one per-step
  download).
- The Python `_validate_state` at admission and export: 3n owned copies
  plus `all_finite` (native) and a pure-Python `any(x < 0 ...)` generator
  over n floats (`_byte_lm_impl.py:273`), which is most of the 6.8 s. It is
  one-time per crossing, and the semantics stay; the implementation lane may
  replace the generator with a native `_buffer` helper `any_negative_f32`
  (a comparison scan, not arithmetic) to make an export cost about 0.5 s
  instead of 7 s. Not required for correctness.
- The `_require_same_bits` mirror admission: no equivalent and none needed;
  it compared a host copy that no longer exists (section 1).

## 7. Gates before any default flips

All bitwise. Fixture A is the small native fixture of
`tools/byte_lm_session_check.py` (B2/L7/DM24, 3 layers, V513); fixture B
is the control shape (20,453,376 parameters, B1/L2048/V8192); the target
shape is RUN OWED on the H100 for G5 and for one pass of G1. The stateless
reference arm and the resident arm run in one process where DEVIATION 2494
allows it; if the second-context hang is still open at gate time, run the
reference arm under `MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1` (DEVIATION 2513) or
in its own subprocess and compare sha256 witnesses, and record which.

| gate | procedure | pass criterion |
|---|---|---|
| G1 multi-step equality | N = 8 steps, same ids sequence, resident `step_result='lean'` versus stateless. After every step: loss bits equal; `export_gradients()['flat_gradients']` bytes equal to the reference `flat_gradients`. After step 8: `export_state()` parameters, m, v, flags, completed_steps equal to the reference `state_dict()`. Include one `evaluate()` between steps 4 and 5 with `export_state()` before and after it. A second resident run with NO intermediate exports must end in the same state (exports are read-only). | every comparison equal; eval export before == after; sha256 of loss, gradients, parameters, m, v at fixture B equal to run 2's `result.json` witnesses (`bench/results/e1g/2026-09-10_233303-nvidia/remote/lm-step-memory/control/`) on the same GPU |
| G2 checkpoint continuation | resident to k = 3, `export_state()`, open a NEW trainer with `load_state_dict` (and at fixture A also `from_checkpoint_bytes`), continue to 8; also `close()` at k = 3 on the original and continue on it | both continuations equal the uninterrupted resident run and the stateless reference at step 8, bitwise, plus the gradient of step 4 exported from each |
| G3 deliberate mutation | after step k: `s = export_state()`; write `+0.25` into `s['parameters'][0]`, `-1` into `s['v'][0]`, `1` into `s['flags'][0]`; `g = export_gradients()`; write `7` into `g['flat_gradients'][0]` and into one named tensor; mutate the `ids` array after the call | step k+1 loss, gradients and state equal the reference's step k+1; `load_state_dict(s)` is refused (negative v) and the live session is untouched (a following step still equals the reference) |
| G4 failed-update controls | a fault-injection build flag `-D MOJOLEARN_BYTE_LM_FAULT_INJECT=1` (compiled only for the gate binary, like the sabotage arms; production builds contain no such code) enabling `MOJOLEARN_BYTE_LM_FAULT=` `loss_nonfinite` (NaN into `ce_loss` after L13), `grad_nonfinite` (NaN into `grad[0]` after `pack_grads`), `opt_refuse` (NaN into `m_state[5]` AFTER the shadow copy, before `identical_optimizer_step`), `after_nonfinite` (inf into `v_state[3]` after the update kernel), `after_negative` (`-1.0` into `v_state[3]` after the update kernel), `python_post` (the fake-binding `wrong_step` path in the host tests, and a monkeypatched post-check in the native tool). For each: `before = export_state()`, one faulted step (must raise with the named message: `"byte LM: nonfinite loss at 0"`, `"byte LM: nonfinite gradients at 0"`, `optimizer: NaN in exp_avg at flat index 5 ...`, `"byte LM: nonfinite second moments at 3"`, `"byte LM: negative second moment"`), then `after = export_state()`, then the fault off and one good step | `after == before` bytewise including flags and `completed_steps`; `export_gradients()` refused after the faulted step; the good step equals the reference's step k+1; the message text of the device refusal equals the host oracle's message on the same planted List (loss_check and optimizer_check clause (f) extended by one string equality) |
| G5 timing | `tools/lm_step_memory_probe.py --target --resident-lean --component-timing` on the H100, three untimed steps and one timed | median complete step (train_step wall at the binding boundary) under 2.0 s; the timed step's `component_timing_ms` contains none of the removed phases of 1.2 and does contain `step.shadow_copy`, `step.ce_refuse_scan`, `step.opt_refuse_scan`, `step.validate_after_scan`, `step.validate_grads_scan`; the untimed sha256 witnesses equal run 2's target `result.json` |
| G6 existing suites | the host suite (112 tests, updated per section 8), `tools/byte_lm_session_check.py` (updated), the three 88-array comparisons of the session handoff, loss_check and optimizer_check with their sabotage arms | all pass; the 88-array comparison prior-versus-resident is the one that proves the shared step body did not change the stateless bytes |

## 8. File-by-file plan, in landing order

Each step is independently gate-able and lands before the next. Line
estimates are for the diff.

| # | file | change | lines | gate |
|---|---|---|---:|---|
| 1 | `core/device_scan.mojo` (new), `core/device_scan_check.mojo` (new), `transformer/impl/llama/fused_attention.mojo` (re-import only) | move `NONFINITE_NONE`, `nonfinite_partial_kernel`, `device_first_nonfinite`; add `negative_partial_kernel`, `device_first_negative`, `device_classify_nonfinite`, optional `DeviceScanScratch` | +140, +90, -60/+3 | new check: NaN, inf, negative planted at index 0, last, and two places (first wins), against host loops; existing transformer gates unchanged |
| 2 | `training/checks/loss_oracle.mojo`, `training/checks/loss.mojo` | `ce_nonfinite_message`, split `ce_refuse_inputs` into `ce_refuse_shape` + `refuse_nonfinite` + `ce_refuse_targets` (same order); `ce_refuse_device_inputs` on the device; timer rename | +40, +50/-25 | loss_check clause (f) with the message equality of G4 |
| 3 | `training/checks/optimizer_oracle.mojo`, `training/checks/optimizer.mojo` | `opt_nonfinite_message`; `opt_refuse_device_inputs` as four device scans; timer rename; docstrings' OWED items 3 marked paid | +25, +45/-40 | optimizer_check clause (f) with the message equality |
| 4 | `training/byte_lm.mojo` | `shadow_*`, `flags_before`, `shadow_valid`, `shadow_step`, `grad_step` in the structs; `byte_validate_device_state`; `_byte_step_device` as the ONE step body (forward, CE, backward, pack, grad scan, shadow copy, optimizer, device validate_after) used by both `byte_train_step` (mirrors around it, unchanged bytes) and the new `byte_train_step_resident -> ByteLeanResult(loss, completed_steps, flags)`; `byte_rollback`; `byte_eval_loss_resident` without the 3n download; timers `step.shadow_copy`, `step.validate_grads_scan`, `step.validate_after_scan` | +260/-60 | the 88-array prior-versus-resident comparison (stateless bytes unchanged); a native resident lean gate at fixture A: 8 steps equal to `byte_train_step`'s capture bytes |
| 5 | `bindings/_mojolearn_byte_lm.mojo` | `byte_lm_session_open(session, [p, m, v, flags], params, shape)`, `byte_lm_session_step(session, [ids, out_loss, out_flags], params, shape) -> step`, `byte_lm_session_eval`, `byte_lm_session_export_state(session, [p, m, v, flags])`, `byte_lm_session_export_gradients(session, [grad])`, `byte_lm_session_rollback(session)`, `byte_lm_session_info(session) -> (completed, grad_step, usable)`; a slot-table form of `_validate_addresses`; the scalar admission block kept per step; `byte_lm_session_run` kept for the transition; pinned-to-Python export copies via `copy_f32`, no Lists | +240 | `tools/byte_lm_session_check.py` extended: open, 8 steps, export equality against the stateless arm, rollback controls of G4 (needs step 7's fault flag) |
| 6 | `python/mojolearn/_byte_lm_impl.py`, `python/mojolearn/language_model.py` | `step_result` keyword; `_state` arrays `None` while a session is open; `_open_session` (admission), `_run_impl` lean branch, `export_state`, `export_gradients`, `export_checkpoint`, `close` exporting first, `_lost_at`; `_run` except path calls rollback; `run_metadata` reports `step_result` and `last_export_step`; the numpy-free docstring updated | +220/-40, +4 | host suite (step 7) |
| 7 | `python/mojolearn/tests/test_byte_lm_surface.py` (FakeByteLM gains the six session entries and a fake device state so exports and rollback are observable), `python/mojolearn/tests/test_byte_lm_session.py` (lean result keys; export isolation; rollback on `wrong_step` and `nonfinite_after_write` leaves state equal; close exports; lost-session error; `resident=False` refuses `'lean'`), `tools/byte_lm_session_check.py`, `training/byte_lm.mojo` fault-injection arm behind `MOJOLEARN_BYTE_LM_FAULT_INJECT` | +120, +130, +60, +40 | G3, G4 at fixture A; G1, G2 at fixture B |
| 8 | `tools/lm_step_memory_probe.py` (`--resident-lean`, sha256 of exports at the end and per step under `--witness-every-step`), `tools/byte_lm_real_text_capture.py` (:173, :194 read `flat_gradients`: call `export_gradients()`), `tools/byte_lm_session_bench.py` (:57), `tools/wp67_lm_surface.py` (:22), `tools/byte_lm_runtime_numerical_check.py` (:112-137), `tools/byte_lm_lifetime_diag.py` (:200; the lifetime lane's file, coordinate) | consume `export_gradients()`/`export_state()` when `step_result='lean'` | +30, +15, +10, +5, +15, +10 | G5 RUN OWED on the H100 |
| 9 | `python/mojolearn/_byte_lm_impl.py` default flip (`step_result='lean'` when `resident=True`), `python/mojolearn/ALPHA_API.md`, `training/BYTE_LM_IMPLEMENTATION.md`, `docs/lanes/HANDOFF_lm_session_2026-09-10.md` ("not yet a minimal-transfer trainer" paragraph), memory brief section 1.6 and rank 7 | docs and the one default | +40 | only after G1 to G6 pass and G5 is filed under `bench/results` |

Existing tests that change: `test_byte_lm_session.py` (all six tests keep
passing with the fake's new entries; `test_native_and_python_failures_discard_advanced_session`
becomes "roll back and keep the session", asserting `closed == []` and
`len(created) == 1` after recovery instead of `closed == created`);
`test_byte_lm_surface.py` (`host` fixture only); `tools/byte_lm_session_check.py`
(the mirror-sabotage control `resident._state['parameters'][0] += .25`
becomes G3's export mutation, since `_state['parameters']` is `None` while
open). The 88-array comparison scripts are unchanged.

The first file to touch is `core/device_scan.mojo` (step 1): steps 2, 3 and
4 all depend on it, it has no dependency on the byte LM, and its check gates
it alone.

### What the memory brief's candidates become

- Rank 1 (backward stage reuse, 2.4 GiB device) and rank 3 (CE aliasing,
  0.8 to 1.15 GiB): unchanged; device memory only; independent of this
  design.
- Rank 2 (row-tiled loss/head, 1.3 GiB): its side benefit "shrinks the
  `ce_refuse_device_inputs` host mirror to 2*R*V" disappears, since that
  mirror is gone after step 2; the device scan becomes per chunk (R*V). It
  stays a device-memory candidate and nothing else.
- Ranks 4 and 5 (parameter and gradient views, 0.6 GiB each and 1.3 GB/step
  of D2D): become the largest REMAINING per-step item after this design,
  since the unpack and pack copies ("under 100 ms each") are now a visible
  fraction of a 0.6 s step. Rank 4 interacts with rollback only in that the
  rollback must be a copy, not a handle swap (4.2 item 4 already says copy).
- Rank 6: unchanged.
- Rank 7 (device scans for the two refusals): subsumed by steps 2 and 3.
- The host mirror inventory (section 1.6 of the brief, 17.7 GiB native and
  10.9 GiB Python per step) goes to zero per step; host RSS at the target
  should drop from the measured 21.9 GB to the process plus a one-time
  admission transient (about 3n host copies during import, freed after
  upload) and an export transient of about 2 x 3n.

## 9. Things in the current code that constrain the design as stated

1. `adam_update_kernel` updates in place, so device double-buffering as
   literally stated (two buffer sets, alternate) needs a kernel signature
   change in the optimizer lane's file. The shadow copy of 4.2 gives the
   same atomicity without touching a kernel, at 1.2 ms per step.
2. `device_first_nonfinite` lives in `transformer/impl/llama/fused_attention.mojo`.
   Training code must not import the transformer implementation, so step 1
   moves it to `core/` and edits one import line in the transformer lane's
   file.
3. Both oracles build their refusal messages inline in `refuse_nonfinite`;
   a device path that reports the scan's index cannot call them with a
   one-element List (it would say "flat index 0"). The message helper of
   6.2 is a refactor inside the oracle files, which are the "answer" files;
   it changes no number, and G4's string equality is what makes "same
   message" a gate rather than a promise.
4. `close()` cannot both "release device state" and "resume from retained
   host state" without an export inside `close()`. That export can fail on
   a lost context, in which case the state is lost; today it never is. This
   is the one semantic weakening, and it is explicit (4.2, last paragraph).
5. `_validate_state`'s `_array` forces a copy of every array it validates
   and its negativity check is a Python generator; at export this is about
   7 s at the target. Not a correctness problem, but the "same validation at
   export" costs that until a native negativity helper exists (6.3).
6. The stateless reference in the gates runs N contexts in one process,
   which is the DEVIATION 2494 hang shape on the RTX 4090; the gates must
   record whether they ran under the DEVIATION 2513 keeper or in
   subprocesses.
7. The 2 MiB JSON checkpoint bound means `export_checkpoint` is refused at
   the target shape exactly as `save_checkpoint` is today; `export_state()`
   arrays are the only target-scale checkpoint, as the docstring already
   says.
