# BRIEF: Byte-LM trainer lifetime hang on NVIDIA (DEVIATION 2494)

Source-analysis lane, 2026-09-10. Nothing in this brief was executed; it is
a reading of the source and of the retained WP6/WP7 evidence, plus a bounded
diagnostic harness whose run is OWED to the orchestrator (last section).
No speed, learning or cross-vendor claim follows from this file.

## 1. What the evidence actually shows

Retained under `bench/results/wp6_wp7_2026-09-10/raw/before/identical/`
(RTX 4090, baseline d330a49d, IDENTICAL) and in the external archive
`mojolearn-evidence/wp6_wp7_2026-09-10/qualification.tar.gz`
(`provenance/wp67_lm_surface.py`, `executed-drivers/`, `*-timeout.partial`):

| capture | driver | sequence | result |
|---|---|---|---|
| `surface-byte_lm-0-train.log` | `wp67_lm_surface.py`, `WP67_LM_RESIDENT=0`, 1 step | one stateless call | PASS, 15 arrays |
| `surface-byte_lm-0-eval.log` | same, eval only | one stateless call | PASS |
| `surface-byte_lm-1-train.log` / `-1-eval.log` | resident, one call each | one resident call | PASS |
| `resident-multi.log` (integration) | resident, 2 steps + eval | resident, resident, eval | PASS, 56 arrays |
| `surface-byte_lm-0.log` | `wp67_lm_surface.py`, `WP67_LM_RESIDENT=0`, `WP67_LM_STEPS=2` | **one** stateless `LanguageModelTrainer`, `train_step` twice | printed `training False 0`, `training False 1`, then HUNG in the second call at `_byte_lm_impl.py:457` (`byte_lm_run_configured`) |
| `surface-byte_lm.log` | `packaging/language_model_smoke.py` line 18 | stateless trainer step, then a second (resident) trainer's first step | HUNG in the resident trainer's first call at `_byte_lm_impl.py:452` (`byte_lm_session_run`) |

Two corrections to the handoff wording:

1. The stateless-to-stateless hang does not need a second Python trainer.
   It is the SECOND NATIVE CALL of the SAME `LanguageModelTrainer(resident=False)`.
   Every stateless call creates and destroys its own native context
   (`bindings/_mojolearn_byte_lm.mojo:216`, `:284-285`), so "a new trainer
   context after a stateless call" means "a second `DeviceContext()` in a
   process whose first `DeviceContext()` has already been destroyed".
2. The `byte_lm-*-timeout.partial` files are NOT a locator. The capture
   driver wrote through a buffered stream and the process was SIGTERMed, so
   the small records (flags, ids) were never flushed. The faulthandler stacks
   in the `.log` files are the only reliable location, and both put the
   process inside the native call (Python line 452 or 457 of the baseline's
   `_byte_lm_impl.py`, which are the same lines in the current file).

Reference hashes from the one completed RTX 4090 stateless step (fixture
`batch=1,length=5,d_model=16,n_heads=2,n_kv=1,head_dim=8,intermediate=24,
n_layers=3,vocab_size=257`, `numpy.random.default_rng(19)`, weights
`standard_normal*.02` then ids `integers(0,257,(1,6))`), sha256 of the
little-endian bytes, so the harness below can be compared against them:

    state.parameters after step 1  5f07d63108aab735d66bdae38d508f387726ca447f70198154d0e87e513b6fe4
    state.m          after step 1  0c1b2b285f4c2421009e1ac1382f6af5f87496498b5727243e9db799962a907a
    state.v          after step 1  c33462c1b5d15bcde76a0875757b935117d51b2cd99678be2a87047a897c5660
    flat_gradients   step 1        59ee565645507db79d7f0d260a9b63781a389560b6ef208568e9bb29b69ada5b
    initial parameters             f015cf0095e5f9bcb5870ca2c798260e33c333d30425611e14389d10e9ec745d
    ids                            f02dc7c267f4cd5e955867da7b3d5108acfc7bbcd77e7cf33d60e33a7f5cf61f

## 2. This is a regression window, not the historical stateless behavior

The exact "stateless call, context destroyed, next stateless call creates a
new context" shape ran hundreds of times per process on NVIDIA before:

- 2026-09-03 and the Sep 7 identity grid: `tools/byte_lm_real_text_capture.py --steps 128`
  drives 128 `train_step` calls through one `SmallByteLanguageModelTrainer`
  per process, and on CUDA `resident` defaults to False (line 331). It passed
  on H100 (three-vendor identical training record).
- 2026-09-07 `bench/speed/byte_lm_speed_arm.py` (commit 05fb8d6d, run at
  58d018c4): 5 rounds x 128 `train_step` calls through `byte_lm_run`, one
  process, H100, 1.09 s per round.

At 58d018c4 the binding created `var ctx = DeviceContext()` and
`var trainer = ByteTrainer(...)` as locals inside `with GILReleased`, then
`ctx.synchronize(); _ = trainer^; _ = ctx^`. The current binding does the
same three things in the same order through `ByteLMSession`
(`bindings/_mojolearn_byte_lm.mojo:280-285`). So the destruction ORDER did
not change. What changed between 58d018c4 and the hanging baseline d330a49d
(git log, 12 commits touching these files) is:

- `37b6b223` the session restructure (Optional context/trainer in a struct,
  `ref ctx = session.ctx.value()`, teardown by `session.trainer = None;
  session.ctx = None` at :284-285 while the GIL is released);
- `72f79a18`, `4bf24a22`, `9bf5115a` generalized shapes and the
  `byte_lm_run_configured` ABI (the WP67 driver used a 3-layer, V257 shape
  through it; the Sep 7 runs used the default profile through `byte_lm_run`);
- `28699cc7` backward gradients kept on device
  (`llama_decoder_layer_backward_device`, `training/byte_lm.mojo:581-595`);
- `de4cf235` lazy attention stages (`lean=True`, `modeling_llama.mojo:1249-1275`)
  and the fused attention path (`transformer/impl/llama/fused_attention.mojo`,
  1640 new lines);
- `587a9107` one reused prefill KV cache per trainer (`training/byte_lm.mojo:427`, `:528`).

None of those was run twice in one process on NVIDIA before WP6/WP7
(`docs/lanes/HANDOFF_lm_session_2026-09-10.md`: "CUDA/HIP session builds and
execution remain unqualified"; every other lane in the window qualified on
Apple). The box also differs (RTX 4090 on a RunPod image versus H100), which
reading cannot exclude.

## 3. The exact native sequence per lifetime

Python entry: `train_step` -> `_run` (`python/mojolearn/_byte_lm_impl.py:407-417`)
-> `_run_impl` (:419-481). Host validation and copies (:420-444) precede
every native call; the dispatch is :445-457:

- resident: `byte_lm_session_create()` once (:450), then
  `byte_lm_session_run(session, ...)` (:452);
- default profile: `byte_lm_run(addresses, parameters)` (:455);
- any other shape: `byte_lm_run_configured(...)` (:457).

Native (`bindings/_mojolearn_byte_lm.mojo`):

- stateless entry points build a LOCAL `var session = ByteLMSession()`
  (:333, :341) and call `_byte_lm_run(..., session, retain=False)`;
- `_byte_lm_run` validates and copies every input on the host (:161-200),
  then inside `with GILReleased(Python())` (:213): `session.ctx = DeviceContext()`
  (:216), `session.trainer = ByteTrainer(session.ctx.value(), ...)` (:217),
  `ref ctx = session.ctx.value()` (:219), the reuse admission when a context
  exists (:220-239: three full downloads and bit compares), the step or eval
  (:240-268), host validation of the outputs (:269-279), `ctx.synchronize()`
  (:280), and for `retain=False` the teardown `session.trainer = None;
  session.ctx = None` (:284-285) STILL INSIDE the GIL-released block;
- on any error inside that block (:286-288) the teardown is skipped; the
  local session's `__deinit__` (:56-59) then drops trainer and context
  after the GIL is re-acquired and WITHOUT a synchronize;
- publication to the caller's output arrays happens after the block (:293-302).

`ByteTrainer.__init__` (`training/byte_lm.mojo:409-435`) allocates
`ByteBuffers` (45 buffers, :264-337, each `_zeros`/`_upload` drains the
stream: `training/checks/train_loop.mojo:1005-1039`), a rope table, a KV
cache, and per layer 9 weight buffers plus lean forward and backward stage
structs (about 30 buffers each). For the WP67 3-layer fixture that is
roughly 250 `DeviceBuffer`s per context, all freed at :284. Per MAX docs,
`DeviceBuffer.__deinit__` "schedules an owned buffer free using the stream in
the device context; the actual deallocation may occur asynchronously", and
`DeviceContext.__deinit__` at zero references "releases the underlying
resources, including any cached memory buffers and compiled device
functions".

Resident lifetime: the same code with `retain=True`; the session lives in a
Python-owned Mojo object. The context dies only in `close()` (:61-68:
`synchronize()` then `trainer = None; ctx = None`, GIL HELD, called from
`_release_session` :369-374) or in `__deinit__` (:56-59, no synchronize)
when the Python object is collected. `load_state_dict` (:376-383) and any
failed call (`_run` :410-417) go through `_release_session`.

Differences between the two paths at teardown/recreation, exhaustively:

| | stateless (`retain=False`) | resident |
|---|---|---|
| where the context dies | :284-285, inside `GILReleased`, right after :280 | `close()` :61-68 with the GIL held (explicit `synchronize()` first), or `__deinit__` :56-59 |
| what dies with it | the whole trainer, ~250 buffers, on the SAME call | same objects, on a LATER call or at collection |
| error path | :286-288 skips teardown; `__deinit__` :56-59 does it unsynchronized after `raise` | `_release_session` calls `close()` (synchronizes) |
| next call | new `DeviceContext()` + new `ByteTrainer` + all kernel loads (:216-217) | reuse (:220-239), three downloads + bit compares |
| Python-side state | none (a Python int comes back) | `_native_session`, `_session_binding` (:323-324, :449-451) |

Nothing else differs. Python holds no native handle for stateless calls; the
only module-level cache is `_backend.binding` (the extension module object,
`python/mojolearn/_backend.py:1060`), and the Mojo tree has no module-level
`var` (grep over every non-archive `.mojo` file), no cached compiled
functions, streams or events (`compile_function`, `DeviceStream`,
`DeviceEvent` do not appear in the trainer's lanes), and no `__deinit__`
other than `ByteLMSession` and an unrelated forest model.

## 4. Candidate causes, ranked by evidence

1. **A second `DeviceContext()` after the first context of this process was
   released with ~250 stream-ordered buffer frees and pinned host frees
   pending, plus the new kernels' module unloads** (`_mojolearn_byte_lm.mojo:284-285`
   then `:216` on the next call; MAX runtime internals, closed source).
   For: every hanging sequence has exactly this shape (create #2 after
   destroy #1: stateless x2, stateless then resident) and every passing one
   does not (resident x2 + eval; one call per process). Against: the same
   order passed 128 to 640 times per process on H100 at 58d018c4, and
   `_mojolearn_trees` / `_mojolearn_mamba` create a context per call on
   NVIDIA today. So the create-after-destroy shape is necessary but not
   sufficient; something in the window of section 2 (or the box) changes
   what the first context leaves behind. The harness separates the
   sub-questions: `resident_then_stateless` (second context while the first
   is ALIVE), `resident_close_reopen` (destroy with the GIL held and an
   explicit `synchronize()` in `close()`), `stateless_x2_gc_pause` (a 3 s
   pause before the second creation), `stateless_x2_default_profile` (the
   Sep 7 path on the current binary), and on a hang: whether the child's CPU
   time advances (spinning in the driver or a Mojo loop) or not (blocked on
   a futex), `nvidia-smi` utilization (a kernel still running versus an
   idle device), and a `gdb`/`py-spy --native` stack if either tool exists.
2. **A trainer-lane change in the window that leaves work or a resource
   attached to context #1 which only bites at the next creation**: the fused
   attention kernels (`fused_attention.mojo`, block-wide `barrier()` inside
   reduction and tile loops at :275-284, :350-359, :480-560, documented as
   reached by every thread at :593),
   the on-device backward (`byte_lm.mojo:575-598`), lean stages that
   reallocate fields (`modeling_llama.mojo:1277-1288`), the reused prefill
   cache (`byte_lm.mojo:528`). For: this is the code that did not exist at
   58d018c4. Against: `ctx.synchronize()` at `_mojolearn_byte_lm.mojo:280`
   completed in call 1 (the driver printed `training False 1` after the
   first result validated), so no kernel of call 1 was still running on the
   context's stream when it was released. `stateless_x2_default_profile`
   hanging too would point here (shared code) rather than at the
   3-layer/V257 shape or the `byte_lm_run_configured` ABI.
3. **Teardown while the GIL is released** (:281-285) racing something that
   needs the GIL or the main thread (Python-owned memory the runtime frees
   through a callback, a finalizer that re-enters Python). For: the resident
   `close()` path, which passed in `resident-multi` only because it was never
   followed by a second creation, holds the GIL. Against: 58d018c4 also tore
   down inside `GILReleased`. `resident_close_reopen` answers this directly.
4. **Not a deadlock but a stall longer than the capture's timeout**: kernel
   reloads for the second context (PTX JIT if the sm target did not match the
   device exactly), repeated in every later call. For: the WP67 build target
   for that 4090 is not recorded next to the log. Against: the first call
   loaded the same kernels within the budget. The harness records CPU
   advance and repeated stacks at intervals; a stall shows different stacks
   and advancing CPU, a deadlock shows the same stack and no advance.
5. **Buffer-freed-at-last-use hazards** (`[[mojo-buffer-freed-at-last-use]]`)
   in the trainer path: `modeling_llama.mojo:716` creates a sub-buffer view
   whose last use is the copy at :717, before the `synchronize()` at :718
   (only when `n != len(buf)`, which the trainer's exactly-sized buffers
   avoid); `train_loop.mojo:782-785` keeps its view alive;
   `byte_lm.mojo:509-510` keeps the pinned id buffers alive.
   Nothing found that releases a buffer under in-flight work on the trainer
   path. Low; listed because it is the class of bug the memory names.
6. **Python-side references** (a global dict or module cache holding a
   native object from context #1): excluded by reading (:323-324, :385-391,
   `_backend.py:1060-1085`); stateless calls return a Python int only.
7. **The box** (RTX 4090 + that RunPod image's driver, versus H100): cannot
   be excluded by reading. The RUN OWED below rents an L40S (sm_89, the same
   architecture family as the 4090); a pass there does not clear the 4090
   until the same script runs on one.

What reading established: the exact sequence and ordering of both paths, the
absence of Python or Mojo global state, that the destruction order is the
same as the last known-good NVIDIA run, that the hang needs
create-after-destroy and is not reproduced by resident reuse, and the change
window. What reading could NOT establish: which side of `DeviceContext()`
the process stops on (creation, first allocation, first launch, first
synchronize), whether the process spins or blocks, and whether the trainer
changes or the session restructure or the box is the trigger. Those are the
harness's job.

## 5. The harness

`tools/byte_lm_lifetime_diag.py` (parent never imports mojolearn):

- one subprocess per case, `--deadline` seconds each (default 120), the
  child under `MOJOLEARN_NUMERIC_MODE=identical` and `faulthandler`
  (periodic dumps every `min(30, deadline/3)` s and on `SIGUSR1`);
- on timeout: `SIGUSR1`, 2 s, then `/proc/<pid>/task/*` state and wchan,
  two CPU-tick samples, three `nvidia-smi` utilization samples, `py-spy
  dump --native` or `gdb thread apply all bt` when present, then `SIGKILL`
  of the child's process group;
- each child rewrites its progress record after every event (`<case>.json`),
  so a hung case still says which native call it entered and with what
  `completed_steps`;
- completed steps record sha256 of `loss`, `flat_gradients`, and the
  committed `parameters`, `m`, `v`, `flags`;
- `summary.json` holds per-case status, `first_step_equality` (every case's
  step 1 must be bit-equal), `second_step_equality` (stateless x2, resident
  x2, close/reopen, restore, mismatch recovery, gc pause, launch blocking
  must agree on step 2), and the mixed cases' second trainer against
  `stateless_x1`.

Cases, in run order: `stateless_x1`, `stateless_x2`, `stateless_then_resident`
(the smoke's order), `resident_x2` (plus eval), `resident_then_stateless`,
`resident_close_reopen`, `restore_then_step` (`save_checkpoint` then
`from_checkpoint`, second context), `failure_recovery` (a wrong-shape ids
buffer refused before native, then a good call), `resident_mismatch_recovery`
(host mirror altered, native admission refuses at
`_mojolearn_byte_lm.mojo:234`, session discarded through `_run` :410-417,
`load_state_dict`, then a step on a new context), `stateless_x2_default_profile`
(B2/L32/DM32 through `byte_lm_run`, the Sep 7 path), `stateless_x2_gc_pause`,
`stateless_x2_launch_blocking` (`CUDA_LAUNCH_BLOCKING=1`).

Fixture: the WP67/smoke shape and `default_rng(19)` recipe when NumPy is
importable (then step-1 hashes are comparable to section 1); otherwise a
pure-Python deterministic fixture, flagged in `record.fixture.source`.

`tools/byte_lm_lifetime_diag.sh` (POSIX sh, dash-checked) is the gemm-leg
EXTRA body: reads the GPU's compute capability (or `MOJOLEARN_GPU_ARCHS`),
builds ONLY `bindings/build_byte_lm.sh` under IDENTICAL into
`python/mojolearn/identical/` (moving any prior `.so` aside with its hash),
reads mode/vendor/profile back from the binary, runs the harness with
`pixi run python3`, and leaves everything under
`/root/gemm_leg_out/byte-lm-lifetime/` (`build.log`, `binding_sha256.txt`,
`binding_readback.txt`, `harness.log`, `status.txt`, `cases/*.json`,
`cases/*.result.json`, `cases/*.pystack.txt`, `cases/*.log`,
`cases/summary.json`). Worst case about 10 min build plus 12 x 120 s.

`tools/byte_lm_session_check.py:58` does `resident._state['parameters'][0] += np.float32(.25)`
on a `mojolearn.Array`, which has no `__setitem__` since the NumPy-free
integration (36d48b07); that control now raises `TypeError` before reaching
the native mismatch check. The harness replaces the Array instead
(`case_resident_mismatch_recovery`). The session check needs the same fix;
not edited here (outside this lane's write set).

## 6. RUN OWED (orchestrator, one light thing at a time)

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_lifetime_diag.sh \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --source-ref <sha> \
        --gpu "NVIDIA L40S" --rent --minutes 60

`<sha>` must be a commit that contains `tools/byte_lm_lifetime_diag.py` and
`tools/byte_lm_lifetime_diag.sh` (both uncommitted at the time of writing;
the leg archives from a ref, not the worktree). The extra runs after the
GEMM device check and card on the same lease. Optional knobs travel in the
environment of the box only if the body sets them; edit the defaults at the
top of the `.sh` if 120 s per case is too short for the L40S build box.

What to read when it comes home (`bench/results/.../remote/`):

1. `extra.log` and `byte-lm-lifetime/status.txt`: `build_exit=0`,
   `readback_exit=0` with `vendor cuda`, `harness_exit`.
2. `byte-lm-lifetime/cases/summary.json` -> `hung`, `failed`, `passed`.
   Expected from the retained evidence: `stateless_x1`, `resident_x2` pass;
   `stateless_x2` and `stateless_then_resident` hang. The discriminators:
   - `resident_then_stateless` passes and `stateless_x2` hangs: a second
     context while the first is alive is fine; the hang needs the first
     context RELEASED. Candidate 1 stays first.
   - `resident_close_reopen` passes and `stateless_x2` hangs: the GIL-held,
     explicitly synchronized `close()` teardown avoids it; candidate 3 rises
     and the fix is to move the stateless teardown out of `GILReleased`
     (or through `close()`), then re-run.
   - `stateless_x2_default_profile` hangs: the regression is in code shared
     with the Sep 7 path (session restructure, fused attention, device
     backward, lean stages, prefill cache); bisect those five commits on the
     same box. Passes: the 3-layer/V257 configured path is the trigger.
   - `stateless_x2_gc_pause` passes: a race with deferred teardown.
   - `stateless_x2_launch_blocking` changes the outcome or the stop point:
     asynchrony is involved.
   - `restore_then_step`, `failure_recovery`, `resident_mismatch_recovery`:
     the gate in the handoff (restore and failure recovery), and whether a
     failed session's unsynchronized `__deinit__` (:56-59) behaves
     differently from :284-285.
3. For each hung case, `cases/<case>.result.json`:
   `hang_diagnostics.cpu_ticks.advancing` (true = spinning, false = blocked),
   `hang_diagnostics.nvidia.samples` (GPU utilization during the hang),
   `hang_diagnostics.native` (`gdb` or `py-spy` output; the top frames
   under `_byte_lm_run` name the runtime call: context creation, buffer
   creation, module load, or stream synchronize), `python_stacks` (should
   show `_run_impl` line 452 or 457 as in the retained logs), and
   `record.events` (the last event is `native_call_begin` with the
   `completed_steps` that call was given).
4. `first_step_equality.fields.*.bit_equal` all true, and
   `second_step_equality` true across every case that completed step 2.
   A false there with everything passing is a separate finding (state
   bits differ by lifetime) and blocks the device-owned step API.
5. Step-1 hashes against section 1 (only when `record.fixture.source`
   starts with `numpy`): equal means the L40S reproduces the 4090's bits;
   different is a cross-box identity finding to record, not to explain here.

If nothing hangs on the L40S, the next run is the same command with
`--gpu "NVIDIA GeForce RTX 4090"` before concluding anything about the
window; the retained hang is on a 4090.

## 7. Files

Created (uncommitted): `tools/byte_lm_lifetime_diag.py`,
`tools/byte_lm_lifetime_diag.sh`, this brief. Nothing else edited.

## Run 1 and 2 results (2026-09-11)

Run 1 (L40S, `bench/results/e1g/2026-09-10_230347-nvidia`): every case
failed in 0.3 s at input validation because the wrapper built only the byte
LM binding and the NumPy-free Python layer takes `all_finite_f32` from the
IDENTICAL base binding. Fixed in a51b6150 (build `bindings/build.sh` first).

Run 2 (L40S sm_89, main a51b6150, `bench/results/e1g/2026-09-10_230815-nvidia`):
all 12 cases PASSED, none hung (stateless x1, stateless x2, stateless then
resident, resident x2, resident then stateless, close/reopen, restore then
step, failure recovery, resident mismatch recovery, stateless x2 with the
default profile, with a GC pause, with CUDA_LAUNCH_BLOCKING). First-step
loss, gradients and state are bit-equal across all eleven compared cases,
second steps across all seven, and the mixed sequences match the pure ones.
Wall 1.7 to 4.9 s per case, 26 s for the harness, 97 s of builds.

What this does and does not say: at this commit, on an L40S, with the WP67
3-layer V257 shape, the second DeviceContext after a stateless call does not
hang. It does not clear the retained RTX 4090 capture; the 4090 run is in
flight (`lifetime-4090`, pod started 03:16Z) and is the column that matters
before any conclusion.
