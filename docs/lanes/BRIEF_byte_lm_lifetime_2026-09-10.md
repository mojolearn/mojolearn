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

## Run 3: the hang REPRODUCES on the RTX 4090 (2026-09-11 03:18Z to 03:39Z, `bench/results/e1g/2026-09-10_231543-nvidia`)

Same commit as run 2 (a51b6150 plus the wrapper fix), same sm_89 build
recipe, Mojo 1.0.0 (ed45d567), driver 580.159.04 with CUDA 13.0 (the L40S
had 580.159.03, CUDA 13.0), host kernel 6.8.0-117 (L40S: 6.17.0-29).

| case | L40S | RTX 4090 |
|---|---|---|
| stateless_x1 | pass | pass |
| resident_x2 | pass | pass |
| resident_then_stateless | pass | pass |
| stateless_x2 | pass | HUNG |
| stateless_then_resident | pass | HUNG |
| resident_close_reopen | pass | HUNG |
| restore_then_step | pass | HUNG |
| failure_recovery | pass | HUNG |
| resident_mismatch_recovery | pass | HUNG |
| stateless_x2_default_profile | pass | HUNG |
| stateless_x2_gc_pause | pass | HUNG |
| stateless_x2_launch_blocking | pass | HUNG |

Every hung child is BLOCKED, not spinning: CPU ticks frozen for the whole
deadline, GPU at 0 percent, 398 MiB, 210 MHz idle clocks. The Python stack
is inside the second native call (`_byte_lm_impl.py:457`). No native stack
(neither gdb nor py-spy on the pod image). CUDA_LAUNCH_BLOCKING and a GC
pause change nothing. The three passing cases are exactly the ones that
never create a DeviceContext after a previous one was destroyed:
resident_x2 reuses one context; resident_then_stateless creates the second
while the first is alive (and never creates a third). Every hung case
creates a context after the process has destroyed one.

So the shape is: on this box, a DeviceContext created after another was
destroyed in the same process never returns from its first use (or from
creation). Not attributed further without a native stack; it is either the
MAX runtime's destroy/recreate path on GeForce with this host kernel, or
our teardown leaving something (a stream, a module, a pinned free) that the
next context waits on. Discriminating control OWED: the same two-context
sequence through a binding that is NOT the byte LM (two KMeans or
ExtraTrees fits in one process), on the same box. If that hangs too, it is
the runtime on that box and every estimator is affected there; if it does
not, it is the byte LM's teardown.

Mitigation to test (not the default until the control says which it is): a
process-lifetime context keeper, so a live context always exists when the
next one is created, which is the exact condition of the passing cases.

## DEVIATION 2513: controls through other bindings, and an opt-in context keeper (2026-09-11, source only)

Nothing in this section was executed. It adds three CONTROL cases and two
mitigation variants to the harness, one switch-guarded change to the byte LM
binding, and the trees build to the wrapper. The run is OWED below.

Line numbers first, because the ones cited above are stale: DEVIATION 2499
(b3d4f3e0, the phase timers) landed after this brief was written, and this
section adds 53 lines. In the current `bindings/_mojolearn_byte_lm.mojo`
the per-call context is created at :296 (`session.ctx = DeviceContext()`,
was :216 then :244) and the stateless teardown is :382-383
(`session.trainer = None; session.ctx = None`, was :284-285 then :330-331).
Everything else in section 3 still reads the same.

### The controls

Every hung case in run 3 creates a `DeviceContext` after the process has
destroyed one. The byte LM is not the only binding that does that:

- `kmeans_fit_binding` (`bindings/_mojolearn.mojo:421`) creates
  `var ctx = DeviceContext()` inside `with GILReleased`, and it dies at the
  end of that block. The base binding does NOT cache its context: every
  `KMeans.fit` (`python/mojolearn/cluster.py:161`) is one create-use-destroy.
- `et_classifier_fit_binding` (`bindings/_mojolearn_trees.mojo:338`) does
  the same per fit. The only process-wide state in that binding is
  `ET_EXPORTS` (`:72`, a `std.ffi._Global` registry) and it holds a HOST
  `FitResult`, no device object; the Python side copies the arrays out and
  calls `forest_export_release` before `fit()` returns
  (`python/mojolearn/_forest_protocol.py:265-310`).

So two fits in one process through either binding is exactly the
create-after-destroy shape, with none of the byte LM's code. The three
cases, each its own subprocess with the same deadline and hang diagnostics:

| case | sequence | binding(s) |
|---|---|---|
| `control_kmeans_x2` | `KMeans(n_clusters=4, numeric_mode='identical').fit` twice, 256 x 4 | `_mojolearn` only |
| `control_extratrees_x2` | `ExtraTreesClassifier(n_estimators=8, max_depth=6, numeric_mode='identical').fit` twice, 512 x 8 | `_mojolearn_trees` (plus `_mojolearn` host helpers) |
| `control_kmeans_then_bytelm` | one KMeans fit, then one stateless byte LM `train_step` (fresh trainer, `completed_steps=0`) | `_mojolearn`, then `_mojolearn_byte_lm` |

The two pure controls never import the byte LM binding (`run_child` skips
the byte LM witness for `CONTROL_ONLY`); they record their own binding's
file, sha256, vendor and compiled mode instead (`record.control_binding`).
Inputs are a pure-Python `random.Random(2513)` fixture through `frombytes`,
so the controls do not depend on NumPy. Small data on purpose: this is a
lifetime question, not a timing one; the 1M-row floor is for tree timing.
Each control hashes both fits' outputs (centroids, labels, inertia; the five
forest arrays) and `summary.json.control_fit_equality` says whether fit 1
and fit 2 are bit-equal, which under IDENTICAL with one seed they must be;
a difference there is a separate finding, not this lane's.

What each outcome means, read together with `stateless_x2` on the same box:

| `control_kmeans_x2` | `control_extratrees_x2` | `control_kmeans_then_bytelm` | reading |
|---|---|---|---|
| pass | pass | pass | create-after-destroy is fine for the other bindings AND for the byte LM's first context after a base-binding context died: the hang needs the byte LM's OWN first context to have been destroyed. The byte LM's teardown (what its ~250 buffers, fused-attention modules, pinned frees or the session restructure leave behind) is the trigger. Bisect the five commits of section 2 on this box; the keeper is a workaround only. |
| pass | pass | HUNG | the byte LM's context CREATION or first use is the sensitive side (a base-binding context died first and the byte LM's first context still hung); still byte LM specific, but the fault is in what the byte LM does on a fresh context after any destroy (kernel loads, first allocations), not in its own teardown. |
| HUNG | HUNG | HUNG | the runtime on this box: any second `DeviceContext` after a destroy blocks, every per-call estimator is affected there, and the byte LM is only where it was noticed. The keeper then belongs in the BASE binding (or a process-wide runtime hook), not in the byte LM; that is a follow-up lane, not this one. |
| HUNG | pass (or the reverse) | any | the two per-call bindings differ in what their first context leaves behind (kernel count, pinned host memory, the export registry); record which and compare its teardown with the byte LM's before attributing. |

The passing controls are only meaningful if `stateless_x2` still hangs in
the same run (the binary changed: the keeper code is in it, switch off); a
run where nothing hangs says the box or image changed, not that anything was
fixed.

### The keeper (mitigation, OFF by default)

`bindings/_mojolearn_byte_lm.mojo:73-112` and `:250-252`, `:289-296`:

- `_ContextKeeper` (:73) holds `Optional[DeviceContext]`; `ensure()` creates
  the context once; `active()` reports whether it holds one.
- `BYTE_LM_CONTEXT_KEEPER` (:105) is a `std.ffi._Global[StorageType=_ContextKeeper, name="MojoByteLMContextKeeperIdentical", init_fn=...]`,
  the same construction as `ET_EXPORTS` (`_mojolearn_trees.mojo:72`) and
  `RF_EXPORTS` (`_mojolearn_rf.mojo:266`). The repo has no module-level
  `var` anywhere (section 3 grep), and the memory names module-level `var`
  and buffer lifetimes as traps; `_Global` is the one process-wide slot
  pattern the bindings already use, runtime-owned, created once by name,
  `get_or_create_ptr()` callable with the GIL released (it is an
  `external_call` into the compiler runtime, no Python involved). Its
  storage lives until the runtime tears down its globals at process exit,
  which is after every per-call context of the process, so the keeper is the
  last context standing either way. What reading could not establish: the
  stdlib source is not in this checkout (compiled `.mojoc` only) and
  `_Global` is undocumented, so whether its deinit runs at exit at all, or
  only on explicit destruction, is not confirmed; both are acceptable here.
- `_byte_lm_run` reads `MOJOLEARN_BYTE_LM_KEEP_CONTEXT` once per call
  (:252, a `getenv` in the same host section and cost class as the timing
  switch `ton`). Inside `GILReleased`, ON THE CREATE PATH ONLY (`if not
  session.ctx`), when the switch is `"1"`, it calls
  `BYTE_LM_CONTEXT_KEEPER.get_or_create_ptr()[].ensure()` (:295) and then
  creates the per-call context exactly as before (:296).

It is a SEPARATE context, created first, never the per-call one retained:
the per-call `session.ctx`, the resident session, the reuse admission, the
`synchronize()` and the teardown at :382-383 are untouched, so a keeper run
differs from `stateless_x2` in exactly one thing, "a live context existed
when the second one was created", which is the condition every passing case
in run 3 shares. Off (the default and every existing caller), the only added
work is the `getenv` compare; the keeper branch is not entered and no
`_Global` slot is created.

Reach is verified, not assumed: `byte_lm_context_keeper_active()` (:109,
exported at the end of `PyInit`) returns whether the keeper holds a context
and creates nothing. The wrapper's readback prints it at import (must be
`False`), and the harness records it after every byte LM call
(`native_call_end.keeper_active`); `summary.json.keeper_reached` carries the
last value per keeper case. A keeper case that passes with
`keeper_active=false` did not test the keeper (stale binary or unreached
branch) and is not evidence.

If `_Global` proved unusable on some target, the fallback is a keeper held
in the Python layer: one `LanguageModelTrainer(resident=True)` whose session
is created at first use and never closed for the life of the process, since
a resident session's context is exactly a never-released context
(`resident_then_stateless` passed on the 4090 for that reason). Not
implemented; the Python layer is another lane's write set.

### Harness variants

`stateless_x2_keep_context` and `resident_close_reopen_keep_context` are the
existing two cases run with `MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1` in the child
environment (`CHILD_ENV`), each recording the switch and the keeper
read-back before and after. Both join `SECOND_STEP_GROUP` and
`FIRST_STEP_GROUP`: the keeper must not change a bit, and a second-step hash
that differs from `stateless_x2`'s on a box where both complete is a finding
that blocks making it a default.

Expected readings, 4090, with `stateless_x2` still hanging:

- both keeper variants pass with `keeper_active=true`: the create-while-alive
  condition is sufficient on that box; the keeper is a usable workaround.
- `stateless_x2_keep_context` hangs with `keeper_active` unobserved (the
  first call is where the keeper context is created): the second context
  (keeper then per-call, both in call 1) already hangs, which contradicts
  `resident_then_stateless`, and the difference would be that the keeper
  context did no work; record it, do not explain it here.
- `stateless_x2_keep_context` passes and `resident_close_reopen_keep_context`
  hangs: `close()` (GIL held, explicit synchronize) leaves something the
  keeper does not cover; candidate 3 rises.

### Wrapper

`tools/byte_lm_lifetime_diag.sh` now builds three bindings under IDENTICAL,
each rm-then-build: base (`bindings/build.sh`, as before), trees
(`bindings/build_trees.sh`, new, for `control_extratrees_x2`; a failed trees
build is recorded as `build_trees_exit` and only that control fails), byte
LM (`bindings/build_byte_lm.sh`, as before). The readback prints
`keeper_readback` and `keeper_active_at_import`. The harness's outer timeout
is 3000 s for 17 cases; worst case (every case hangs at 120 s) is 34 min
plus three builds, about 40 min, inside the 60 minute lease with run 3's
2 minute gates and the 600 s fetch reserve.

### RUN OWED (orchestrator, one light thing at a time)

    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/byte_lm_lifetime_diag.sh \
    MOJOLEARN_GPU_ARCHS=sm_89 \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --source-ref <sha> \
        --gpu "NVIDIA GeForce RTX 4090" --rent --minutes 60

`<sha>` must contain this section's three files (`bindings/_mojolearn_byte_lm.mojo`,
`tools/byte_lm_lifetime_diag.py`, `tools/byte_lm_lifetime_diag.sh`; all
uncommitted at the time of writing). The same leg as run 3, same box class,
same arch. If the lease is a concern, the wrapper honors
`BYTE_LM_LIFETIME_CASES` only from the box's environment (edit the default
at the top of the `.sh`); the decisive subset is
`stateless_x1,stateless_x2,control_kmeans_x2,control_extratrees_x2,control_kmeans_then_bytelm,stateless_x2_keep_context,resident_close_reopen_keep_context`.

What to read when it comes home, in addition to section 6:

1. `byte-lm-lifetime/status.txt`: `build_base_exit=0`, `build_trees_exit=0`,
   `build_exit=0`, `readback_exit=0`; `binding_readback.txt` shows
   `keeper_readback True` and `keeper_active_at_import False`.
2. `cases/summary.json`: `stateless_x2` in `hung` (the anchor), then the
   three controls against the table above, then `keeper_reached` (both
   `keeper_active: true`) and the keeper variants' status.
3. `control_fit_equality.*.all_bit_equal` true for both controls;
   `second_step_equality` still true with the keeper cases included.
4. For any hung control, its `result.json` `hang_diagnostics` (CPU ticks,
   nvidia-smi) should look like run 3's (blocked, idle GPU) if it is the
   same hang; a spinning or busy-GPU control is a different stop and is
   recorded as such.

### What would make the keeper the default

Only one of two results, and neither is decided by this lane:

- the controls pass and both keeper variants pass with the keeper reached
  and every hash unchanged: the byte LM is special on that box, and the
  keeper (or a resident-by-default session, which is the same thing with a
  name) can become the byte LM's default for CUDA, AFTER the bisect of
  section 2 says what the byte LM's first context leaves behind, so the
  default is a fix and not a bandage over an unknown; or
- every control hangs and the keeper variants pass: all per-call bindings
  are affected on that box, the keeper is the right shape, and it belongs in
  the base binding (one process-wide slot every binding checks), not in the
  byte LM. That is a follow-up lane with its own deviation number; this lane
  does not touch the base binding.

Any other combination keeps the switch OFF and the finding open. Not
qualified on any vendor; no arithmetic changed on any path.

### Files (DEVIATION 2513, uncommitted)

Edited: `bindings/_mojolearn_byte_lm.mojo` (+53, the keeper, the switch,
the read-back), `tools/byte_lm_lifetime_diag.py` (five cases, controls,
keeper read-back, `control_fit_equality` and `keeper_reached` in the
summary), `tools/byte_lm_lifetime_diag.sh` (trees build, readback lines,
outer timeout, summary rows), this brief. Nothing else, and nothing under
`python/mojolearn/` or `training/`.

## Runs 4 and 5 (2026-09-11, RTX 4090): controls pass, keeper reached and useless, ptrace blocked

Run 4 (`bench/results/e1g/2026-09-10_235415-nvidia`, sm_89, all three
builds green, 17 cases, 1402 s): the three DEVIATION 2513 controls PASSED
(`control_kmeans_x2`, `control_extratrees_x2`, `control_kmeans_then_bytelm`;
both `control_fit_equality` rows bit-equal), the same three byte LM cases
passed as in run 3, the same nine byte LM cases HUNG, and BOTH keeper
variants HUNG with the keeper REACHED (`keeper_reached.*.keeper_active
true`, `keep_context_env 1`). Read against the DEVIATION 2513 table, that
is the first row: create-after-destroy works on this box for the base and
trees bindings, and for the byte LM's first context after a base-binding
context died. The hang needs the byte LM's OWN trainer to have been
destroyed, and a separate live context does not help, which removes the
"create while another is alive" reading of run 3's passing cases (the
keeper IS a live context, and it did not help). What is left: something the
byte LM's teardown leaves behind, or something the byte LM's SECOND context
does that its first does not, on this box. The keeper stays OFF and is not
a workaround.

Run 5 (`bench/results/e1g/2026-09-11_002601-nvidia`, main 3bae97ad,
subset `stateless_x1,stateless_x2,resident_close_reopen`): the wrapper
installed gdb from apt (`gdb=/usr/bin/gdb`), py-spy did not install, and
`gdb -p` on both hung children returned "ptrace: Inappropriate ioctl for
device" (the pod's ptrace is filtered; `/proc/<pid>/task/<tid>/stack` was
EACCES even from the parent as root). So no tool outside the process can
read the native stack on that image. The `/proc` snapshot did give the
shape: 132 threads per hung child, every one in state S with
`wchan futex_wait_queue` except two in `do_poll`, CPU ticks frozen
(362 -> 362 for `stateless_x2`; 372 -> 373 for `resident_close_reopen`, one
tick), Python stack at `_byte_lm_impl.py:488` (`byte_lm_run_configured`) or
`:483` (`byte_lm_session_run`), last event `native_call_begin step2`. A
blocked main thread in a futex inside the second native call, with nothing
running on the device.

## DEVIATION 2518: in-process native stack, and two teardown variants (source only)

Nothing in this section was executed. Three files carry it; the run is
OWED below.

### The in-process native stack (no ptrace)

`tools/native_stack_dump.c` (new) is a shared library whose constructor,
when `MOJOLEARN_NATIVE_STACK_FILE` is set, installs a `SIGUSR2` handler
(`SA_SIGINFO | SA_RESTART`) and writes one "handler installed" line to that
file; when the variable is unset it installs nothing. The handler appends,
per delivery: a header (`pid`, `tid`, `CLOCK_REALTIME` seconds.nanoseconds,
sender pid), the interrupted `pc`/`sp` from the ucontext (x86_64 and
aarch64), `backtrace()` of the receiving thread printed by
`backtrace_symbols_fd` (`module(symbol+offset) [address]`; stripped
modules such as `libcuda.so.1` give `module(+offset)`, which still names
the library), then `/proc/self/task/<tid>/wchan`, `syscall` and `stack`
for the same thread, and returns, so the interrupted futex or condition
wait resumes. Only `write`, `open`, `read`, `close`, `clock_gettime`,
`getpid`, `gettid`, `backtrace` and `backtrace_symbols_fd` run inside the
handler; the constructor calls `backtrace` once so glibc's lazy `libgcc_s`
load happens outside signal context. Two caveats are in the file header:
the wchan/syscall lines are the receiving thread's view of ITSELF while it
runs the handler (they say "running" or `read`), so the interrupted wait is
what the backtrace and `pc` show and the parent's outside snapshot (which
now also reads `/proc/<pid>/task/<tid>/syscall`) is the per-thread kernel
truth; and a thread hung inside the dynamic loader would deadlock in the
unwinder, in which case the header line is written and the backtrace is
not, and the parent still kills the child at the deadline.

The harness (`tools/byte_lm_lifetime_diag.py`) does three things with it:

1. `run_case` gets `--native-stack-lib PATH` and, for the CHILD only, sets
   `LD_PRELOAD=PATH` (prepended to any existing value) and
   `MOJOLEARN_NATIVE_STACK_FILE=<out>/<case>.native_stack.txt`. The parent,
   `nvidia-smi` and `gdb` never load it.
2. `run_child`, before importing mojolearn, records
   `record.native_stack` (`file`, `ld_preload`, `handler_installed`,
   `watchdog_armed`, the main thread's native id). `handler_installed` is
   `signal.getsignal(SIGUSR2) is None`, which is exactly "a C-level handler
   not installed from Python is present". Only then does it start a daemon
   thread that sleeps until `deadline - 10 s` and calls
   `signal.pthread_kill(threading.main_thread().ident, SIGUSR2)` three
   times two seconds apart, recording each send as a
   `native_stack_signal` event (unix time, so the file's headers can be
   matched to the sends). Without the preload the watchdog is never armed:
   SIGUSR2's default disposition would terminate the child before the
   parent's diagnostics.
3. On a timeout the parent sends one process-directed `SIGUSR2` itself
   (the kernel offers it to the main thread first) before the `SIGUSR1`
   faulthandler dump: this is the fallback for a hang that holds the GIL,
   where the child's watchdog cannot run Python. After the kill it reads
   the file into `hang_diagnostics.native.in_process` (`samples` counts
   the sample headers, `installed_line`, `contents`, capped at 400 KB with
   the middle elided) and records `parent_sigusr2`. Every case, hung or
   not, carries `result.native_stack` (`handler_installed`,
   `installed_line`, `samples`); `summary.json.native_stack_reached` lists
   handler/watchdog/signals/samples per case.

A passing case is unchanged by all of this: the library installs one
signal handler and nothing else, the watchdog sleeps and dies with the
process, no signal is ever sent before `deadline - 10 s` (run 4's passing
cases took 1.7 to 5 s), and `stateless_x1`'s step-1 hashes must still equal
run 4's. If they do not, the preload is the first suspect and
`BYTE_LM_LIFETIME_NO_NATIVE_STACK=1` in the wrapper file turns it off.

How to read the samples of a hung case: three headers with the same frames
is a wait (a deadlock or a wait on something that never arrives); frames
that move between samples is a stall (a long operation, not a deadlock);
`samples 0` with `handler_installed true` and three `native_stack_signal`
events means the signal did not reach our handler (the main thread blocks
it, or a runtime handler replaced ours after import), which is recorded as
such. The frames of interest are the ones between `__restore_rt` (the
signal trampoline) and the Python `PyEval` frames: the byte LM binding's
symbols, then MAX runtime and `libcuda.so.1` offsets. Whether the top
non-libc frame is inside the CUDA driver (`libcuda.so.1(+0x...)`), the MAX
runtime, or our own code (a Mojo symbol from `_mojolearn_byte_lm.so`)
decides where DEVIATION 2519 looks; the offset inside `libcuda.so.1` can be
resolved against that driver build's symbols on any box with the same
driver (580.159.04).

### The two teardown variants (`bindings/_mojolearn_byte_lm.mojo`)

Both are read once per call as `getenv` compares next to the existing
`ton` and `keep_context` reads (`:278-279`), OFF by default, and touch
only the stateless (`retain=False`) teardown of `_byte_lm_run`; the
resident session, the DEVIATION 2514 device-owned entries, the reuse
admission and every kernel are untouched. Each prints one witness line
(`byte LM teardown variant: <name>`) when its branch runs, which the
harness greps out of the case log into
`summary.json.teardown_variant_reached`; a variant case that passes
without its witness line did not test the variant.

- (a) `MOJOLEARN_BYTE_LM_SYNC_BEFORE_TEARDOWN=1` (`:409-417`): inside the
  `GILReleased` block, after the existing `ctx.synchronize()` at `:404`,
  the teardown becomes `synchronize(); trainer = None; synchronize();
  ctx = None`. The first drain is the literal request (a second drain of
  an already drained stream, so on its own it should change nothing); the
  second is the one that can differ: `DeviceBuffer.__deinit__` schedules
  its free on the context's stream, so the ~250 frees the trainer's
  release enqueues are drained while the context is still alive, instead
  of being left for `DeviceContext.__deinit__`.
- (b) `MOJOLEARN_BYTE_LM_TEARDOWN_WITH_GIL=1` (`:406` skips the in-block
  teardown, `:426-433` does it): `trainer = None; ctx = None` run after
  the `with GILReleased` block has re-acquired the GIL and before
  publication, the resident `close()` shape (`:77-84`: synchronized, then
  released with the GIL held).

What each outcome means, with `stateless_x2` still hanging in the same run
(the anchor; a run where nothing hangs says the box changed):

| `stateless_x2_sync_teardown` | `stateless_x2_teardown_with_gil` | reading |
|---|---|---|
| pass, witness seen | hang | the undrained buffer frees of context #1 are what context #2 waits on; the fix is the second synchronize in (a) on the stateless path (and in `close()`, which already synchronizes BEFORE releasing but not after), and the bisect of section 2 narrows to what added frees or changed their order (device backward `28699cc7`, prefill cache `587a9107`, lean stages `de4cf235`). |
| hang | pass, witness seen | releasing under a released GIL is the trigger; since `resident_close_reopen` (GIL held in `close()`) ALSO hung in runs 3 and 4, the difference would be that (b) releases in the SAME call as the step, so record it as "GIL-held release in the step call passes, GIL-held release in a later call hangs" and do not attribute further without the stack. |
| hang | hang | neither the drain nor the GIL is the trigger; the native stack (above) is the only remaining datum, and DEVIATION 2519 starts from its top frame. |
| pass | pass | both changes avoid it; (a) is the cheaper explanation (a drained stream also makes the GIL-held release's timing irrelevant); confirm by running (a) alone on the resident path before choosing. |

Both variant cases join `FIRST_STEP_GROUP` and `SECOND_STEP_GROUP`: a
variant must not change a bit, and a step-2 hash differing from
`stateless_x2`'s on a box where both complete blocks making it a default.

### Wrapper

`tools/byte_lm_lifetime_diag.sh` compiles the library on the box with
`cc -shared -fPIC -O1 -g` (first of `cc`, `gcc`, `clang` on PATH; the pod
image has gcc) into `$OUT/native_stack_dump.so`, records
`native_stack_lib=` in `status.txt` (`none (no compiler)` or
`none (build failed)` are status lines, not failures) and passes
`--native-stack-lib` to the harness; `BYTE_LM_LIFETIME_NO_NATIVE_STACK=1`
skips it. The summary print adds the variant witness rows and, for hung or
handler-less cases, the native stack reach row. 19 cases; the outer 3000 s
timeout still covers the worst case (19 x 123 s plus builds).

### RUN OWED (orchestrator, one light thing at a time)

The same leg as runs 3 to 5, same box class, same arch, with a wrapper file
that keeps the run to four cases (the two anchors and the two variants;
each hung case costs its 120 s deadline):

    cat > /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a424f2ae-c0f9-4785-ad1c-22e8f833ba55/scratchpad/byte_lm_lifetime_2518.sh <<'EOF2'
    #!/bin/sh
    export BYTE_LM_LIFETIME_CASES=stateless_x1,stateless_x2,stateless_x2_sync_teardown,stateless_x2_teardown_with_gil
    exec sh /root/mojolearn/tools/byte_lm_lifetime_diag.sh
    EOF2
    MOJOLEARN_RUNPOD_KEY_FILE=~/.mojolearn_runpod_key \
    MOJOLEARN_GEMM_LEG_EXTRA=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/a424f2ae-c0f9-4785-ad1c-22e8f833ba55/scratchpad/byte_lm_lifetime_2518.sh \
    MOJOLEARN_GPU_ARCHS=sm_89 \
    sh tools/gemm_remote_leg.sh nvidia --payload gemm --source-ref <sha> \
        --gpu "NVIDIA GeForce RTX 4090" --rent --minutes 60

`<sha>` must contain this section's four files (`tools/native_stack_dump.c`,
`tools/byte_lm_lifetime_diag.py`, `tools/byte_lm_lifetime_diag.sh`,
`bindings/_mojolearn_byte_lm.mojo`; all uncommitted at the time of
writing). The wrapper file lives outside the repo on purpose (the leg
copies it to `/root/gemm_leg_extra.sh` and retains it as `extra_body.sh`,
as run 5 did).

What to read when it comes home, in addition to the earlier sections:

1. `byte-lm-lifetime/status.txt`: the three `build_*_exit=0`,
   `native_stack_lib=<path> cc=<compiler>` (not `none`).
2. `cases/summary.json`: `stateless_x1` passed and `stateless_x2` hung
   (the anchors); `native_stack_reached.stateless_x2` with
   `handler_installed true`, `watchdog_armed true`, `signals_sent 3`,
   `samples 3` or `4` (the parent's fallback adds one); then
   `teardown_variant_reached.*.reached true` for both variants, and their
   status against the table above.
3. `cases/stateless_x2.result.json` ->
   `hang_diagnostics.native.in_process.contents`: the three samples. Same
   frames three times = a wait; the top non-libc frame names the owner.
   `hang_diagnostics.proc_before.threads.<pid>.syscall` (the main thread's
   number and arguments, from outside) confirms the futex and its address.
4. `first_step_equality` still all bit-equal with the preload in place
   (the preload changed nothing) and `second_step_equality` true for
   whichever variant completed.

### Files (DEVIATION 2518, uncommitted)

Created: `tools/native_stack_dump.c`. Edited:
`tools/byte_lm_lifetime_diag.py` (watchdog thread, preload plumbing,
parent fallback signal, `syscall` in the `/proc` snapshot, two cases, two
summary rows), `tools/byte_lm_lifetime_diag.sh` (library build,
`--native-stack-lib`, summary rows, knob), `bindings/_mojolearn_byte_lm.mojo`
(two switch reads at `:278-279`, variant (a) at `:409-417`, variant (b) at
`:406` and `:426-433`), this brief. Nothing under `python/mojolearn/`,
`training/` or `neighbors/`. Compile-checked: the C file with `cc -c -Wall
-Wextra` (into the scratchpad, Linux-only parts guarded), the harness with
`py_compile` and a stdlib-only smoke of its helpers, the wrapper with
`sh -n` and `dash -n`. The Mojo edit is unbuilt; the RUN OWED builds it.
