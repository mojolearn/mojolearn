# Asks for Modular

Features we wait for instead of working around (CLAUDE.md, "Wait for Mojo and Modular"). One section per ask:
what we need, where it would be used, what we do meanwhile.

## Masked (subset) warp reduction and native grouping on AMD (2026-10-07, lane trees-hist-ideas)

**Need.** Warp-aggregated histogram atomics in their general form: lanes that hit the same histogram cell
(`std.gpu.primitives.warp.match_any`) add their values once. That needs a reduction over an ARBITRARY lane
subset, the group mask `match_any` returns:

- NVIDIA: `redux.sync.add.s32 dst, src, membermask` (sm_80+), or `__reduce_add_sync(mask, v)`. Mojo's `warp`
  module exposes `sum`/`reduce` over the whole warp or a lane group of fixed width, not over a member mask.
- AMD (CDNA, wave64): no hardware match or masked reduce. `match_any` builds for gfx942 (probed 2026-10-07),
  but we do not know whether it lowers to a per-distinct-value loop; its cost on a 256-bin key is the question.
  A documented cost model, or a native lowering, would let us decide.

**Where.** `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo` (`h8_add_point_wave`), and the
same pattern in RF class-count histograms and categorical/CTR counters (low-cardinality bins).

**Meanwhile.** `MOJOLEARN_TREES_HIST_WARP_AGG` aggregates only the whole-wave case (every lane on the same cell:
`warp.broadcast` + `vote` + `warp.sum`), which needs no subset reduce. Partial groups take one atomic per lane,
as before. No hand-written intrinsic or inline assembly.

## DeviceContext memory manager reserves 90% of free VRAM per process on AMD (2026-10-16, amd-investigate)

**What we see.** On an AMD Instinct MI300X VF (ROCm 7.2.4, gfx942), the first non-small allocation of a process makes MAX's
DeviceContext memory manager take one chunk of 90% of FREE VRAM (`hipMalloc` of 184.9 GB, about 4.2 s) plus 1.28 GiB of
pinned host memory. A following process waits about 5 s for that chunk to be released and cleared (ttm delayed delete and
SDMA clear), and a second concurrent process on the same GPU starves: allocations fail, later HIP calls return
`hipErrorUnknown`/`hipErrorNoBinaryForGpu`, and `libamdhip64.so.7.2.70204` segfaults in its allocation-failure path
(`segfault at 8 ... lock xadd`).

**The ask.** A default that does not assume one process owns the GPU: a bounded or growing chunk (the binary already has
`growing_chunk_size_to` and `min_chunk_size`), or documented guidance for libraries that may share a device. Also, the
libamdhip64 null dereference on allocation failure belongs to AMD's CLR (reproducer: two concurrent processes with the
default manager on one VF).

**Where.** `python/mojolearn/_backend.py` `_bound_amd_memory_manager` (all AMD processes); `tools/gemm_remote_leg.sh` (AMD leg).

**Meanwhile.** mojolearn sets `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE=1073741824` on AMD hosts with `setdefault`
(a caller's own setting wins), which MAX supports: fresh processes 0.78 s instead of 5.9 s, concurrent processes pass, no
bit changes (`verify --compare` against NVIDIA: 1747 agree, 0 differ). NVIDIA keeps MAX's defaults.

## Reproducible gfx942 code generation with a cold cache (2026-10-20, lane amd-compile-speed)

**Need.** `mojo build --target-accelerator gfx942` gives the same embedded AMDGPU code objects every time it compiles
the same source with the same flags. Measured at d181d9792 (Mojo cache wiped before every build): `build_mixture.sh`
3 distinct binaries in 5 builds at `-j 2`, 1 in 5 at `-j 1`; `build_tsa.sh` 2 in 4 at `-j 2`, 2 in 12 at `-j 1`. The
differences are inside kernels (gemm_identical, holtwinters): a few register numbers or two swapped instructions, so the
order of the compiler's own work changes its output, even with one worker. NVIDIA and CPU codegen of the same commits
matched in every cold pair (264 of 264 NVIDIA binaries).

**The ask.** Deterministic AMDGPU codegen (no pointer-ordered or thread-ordered iteration in scheduling or register
allocation), independent of `-j`, so two cold builds of one commit are byte-identical as on NVIDIA.

**Where.** `packaging/linux/build_sets.sh` (the gfx* sets' compiler workers, `AMD_COMPILE_JOBS_DEFAULT`),
`tools/release_reuse.py` (gfx rebuild NOTE), `tools/bincache.py` (the cache freezes the first bytes per key).

**Meanwhile.** The release contract is the output digest, checked on the hardware; the binding cache and release reuse
freeze shipped bytes. `tools/amd_compile_variance.sh` measures whether parallel workers change results or only bytes
and classifies each pair (`tools/compare_gpu_builds.py`: REORDERED and SAME-KERNELS are layout, CODE-DIFFERS is this
ask); the AMD worker count follows its verdict. We do not canonicalize or rewrite differing kernel code.

**2026-10-20 result.** The retained [MI325X experiment](../bench/results/amd_compile_variance/2026-10-20/summary.json)
at `6bf016f25eabf29f44e2cfed160d0d62151ce31a`, Mojo 1.0.0 (`ed45d567`), found
equal GMM results across two one-worker and three eight-worker cold builds.
The cross-worker binaries were REORDERED with the same 137 kernels. The
eight-worker default follows that result; this does not resolve the older
kernel-code variance above or establish results for other estimators.
The merged one-worker build also exited 139 after 515.196 seconds, while
eight workers completed in 794.290 seconds. Its crash log is retained with
the experiment; a minimized compiler-crash reproducer remains pending.
The empty merged canonical comparison (zero files on both sides) proves
neither numerical equality nor a compile-time speedup.

## PythonObject attribute assignment and calls retain one extra reference (2026-10-21, lane finish-leftovers)

**What we see.** In a Mojo Python extension built with the supported `std.python` API, four ordinary operations leave one
extra reference on the value after every intentional Python reference is dropped and garbage is collected, so the value is
never freed: direct attribute assignment `owner.value = value`, builtin calls `type(value)` and `list(value)`, and the
builtin method call `owner.append(value)`. The controls `no_op` and `return owner.value` balance (0 extra, collected).
Receiver counts are unchanged. A Python-only control passes all six cases. The source suggests two transfer sites:
`PythonObject.__setattr__` passes an owned value through `steal_data()` to `PyObject_SetAttrString` (which does not steal),
and `PythonObject.__call__` adds `Py_NewRef` around a transferred argument before `PyTuple_SetItem` (which does steal)
(`mojo/stdlib/std/python/object.mojo` at tag `mojo/v1.0.0`, lines 455-467 and 1270-1297).

**Reproducer.** `experiments/python_object_ownership/` (`_mojo_python_ownership.mojo`, `probe.py`, `README.md`,
`result-20261020.json`); imports only `std.python`, `std.python.bindings` and `abort`, never mojolearn.
`pixi run mojo build --emit shared-lib _mojo_python_ownership.mojo -o _mojo_python_ownership.so && python probe.py`:
build exit 0, probe exit 1 with the four failures. Retained source, binary, logs and hashes:
`~/mojolearn-evidence/core-completion-20261020/stdlib-ownership/`.

**Version.** Mojo 1.0.0 (`ed45d567`), macOS 26.5.2 arm64; reproduced under CPython 3.14.7 and 3.12 with the same binary.

**Impact.** Every native owner that stores Python objects on a Python instance or passes them to Python calls leaks them.
`python/mojolearn/tests/test_model_native_lifetime.py` fails for the llama, mamba and mamba2 `CausalLM` families (the caller's
owner and layer list each keep one extra reference after the state is deleted). Object lifetime is one of the conditions
that kept the native LinearRegression/Ridge owner inactive until 2026-10-21 (see below).

**The ask.** A supported release in which `PythonObject` attribute assignment and ordinary calls (including builtin types
and methods) balance their owned references, with the six operation controls above, and the matching error paths, covered
upstream. The unchanged probe must exit 0, then the project lifetime tests must pass.

**Meanwhile.** We wait: no toolchain or standard-library patch, no compensating reference operations in product code, no
Python runtime substitute. Numerical and native-call audits stay valid for their boundaries but do not qualify lifetime.

**Designed around (2026-10-21, lane activate-linear-installer).** The native LinearRegression/Ridge owner (public since
then) never calls `PythonObject.__call__` with positional arguments, `__setattr__`, `__setitem__` or a set literal on its
hot path. `bindings/py_balanced.mojo` calls the documented CPython entries directly with their documented ownership
(`PyObject_Call` on a tuple it owns and releases once, `PyObject_SetItem`, `PyObject_SetAttrString`, `PyObject_Type`,
`PySequence_Contains`; arguments borrowed, results owned). Nothing adds or drops a reference to offset the standard
library. `python/mojolearn/tests/test_native_classical_public.py` checks that the caller's arrays keep their reference
counts, the model and its outputs are collected, and repeated calls allocate nothing lasting. The ask stands: once the
supported operations balance, these call sites can return to the ordinary syntax.

**Shared shells (2026-10-21, lane leak-shared-helpers).** The shared door helpers (`bindings/generation_buffers.mojo`,
`native_json`, `native_transport`, `native_codec`), the SmallMLPTrainer, Sequential, CausalLM, Samba, parallel,
cross-vendor and autograd doors, and every host binding's `operator.index` argument read use the same helpers.
`python/mojolearn/tests/test_native_steady_state.py` holds the caller's input reference counts and the traced bytes per
call flat for `SmallMLPTrainer.train_step`, the byte-LM `train_step`, `Sequential.fit`, `CausalLM.generate` and both
`sha256_hex` doors. Measured on the Mac (Mojo 1.0.0, CPython 3.14), quietest of four 200-call windows after 500 warm-up
calls: SmallMLPTrainer 10.4 KB to 39 B per call, Sequential.fit 3.2 KB to 17 B, CausalLM.generate 59.4 KB and two
references on the caller's ids to 8 B and none, the byte-LM `sha256_hex` one reference on the caller's bytes to none (the
byte-LM `train_step` was already flat). `python/mojolearn/tests/test_model_native_lifetime.py` passes for all three
families. Retaining every array had also hidden reads after a local's last use (Mojo destroys a local at its last named
use): the decode loop read `logits` by address after that point, which showed as nondeterministic sampled tokens once
arrays were freed. Every array read by raw address now stays alive until the reading call returns. ParallelCausalLM's
remote layer states, now freed, release their worker state through a native lifetime owner instead of a Python
`__del__`, which would have run inside the native calls that drop them.

## PythonObject slices release borrowed None defaults (2026-10-22)

**What we see.** Mojo 1.0.0 (`ed45d567`) converts an omitted slice bound or step
using borrowed `Py_None()`, passes it to `PySlice_New` (which borrows its inputs),
then decrements it as if it owned the reference. Each omitted component loses one
`None` reference. The pinned source is
[`python_object.mojo` at `mojo/v1.0.0`, `_slice_to_py_object_ptr`](https://github.com/modular/modular/blob/mojo/v1.0.0/mojo/stdlib/std/python/python_object.mojo#L1736-L1765).
The public [CPython slice API](https://docs.python.org/3/c-api/slice.html#c.PySlice_New)
accepts null defaults, returns a new slice reference and does not steal the bounds.

**Reproducer and impact.** On CPython 3.11, 200 `PythonObject("abcdef")[0:2]`
operations lose exactly 200 `None` references. Two controls balance: 200 calls to
`PySlice_New(start, stop, null)` plus `PyObject_GetItem`, and 200 owned
`PythonObject(None)` constructions. The pinned-toolchain standalone probe exits
0 and retains its source, log and interpreter provenance under
`~/mojolearn-evidence/takeover-0856-20261021/none-refcount-3b5991a57/`
(`slice_probe.mojo`, `slice-probe-pixi.log`). A first invocation without the pixi
stdlib environment failed and is retained separately. Successful public model
loads lost about 139 references each on three frozen binaries, including the
previously accepted `b9f12538c`. The combined `3b5991a57` NumPy 1.26 suite passed
323 assertions but aborted at interpreter shutdown while deallocating `None`;
that run is a failure, not a passing qualification. CPython's immortal-None
versions can hide this defect from a refcount-only test.

**The ask.** Balance all explicit and omitted start/stop/step combinations in a
supported release, including negative/zero steps, failed item access, repeated
calls and interpreter shutdown on supported non-immortal-None Python versions.
We do not modify the toolchain or adjust references to offset the defect.
`bindings/py_slice_native.mojo` instead uses the documented slice/item APIs with
owned bound, slice and result objects, as confirmed by the control above. It is
only ownership machinery; product callers must still admit native receivers and
bounds before calling it. The public repeated archive-load regression covers
successful consumed state and late schema errors in fresh interpreters. This
source repair is pending native build and execution at this checkpoint; the
original shutdown failure and unaffected accepted evidence remain retained.

## sm_120 AOT miscompiles a SIMD lane written under a per-lane guard (2026-10-23)

**What we see.** Binaries built with `--target-accelerator sm_120` (RunPod RTX 5090, driver 580.178.04) return wrong
values for one lane of a SIMD local whose lanes are each written under their own `if` inside a `comptime for`. The same
source is right on sm_89, gfx942, Apple and under JIT on the same GPU (first seen 2026-09-14 in `core/gram_splitk.mojo`).
On 2026-10-23 it changed the tree structure of every Ordered-boosting GBDT fit (5 identity lanes: 8-leaf trees where every
other column grows 64) while the PTX fused-op census for sm_89 vs sm_120 was identical, which points past the PTX at the
sm_120 back end.

**The ask.** Fix lane-masked insert/extract of SIMD values held in registers on sm_120 (or the ptxas path it reaches), and
add a regression test for a guarded per-lane write in an unrolled loop. Until then our kernels keep such scratch in
`InlineArray` or write every lane unconditionally from a selected value (lane/sm120-tree-identity 45b2db146); no compiler
output is patched.

## Runtime worker-thread count: a supported knob, and honoring the cgroup CPU quota (2026-10-24, fix-round2-host)

**What we see.** Every Mojo process starts one runtime worker per logical CPU it sees: 128-144 threads on the RunPod
L4 pod (nproc 128), whose cgroup grants only 13.6 CPUs (`cpu.max` "1360000 100000"). One process alone is fine (a
4-tree forest fit in about 10 s), but a parallel driver's parent plus its worker process, each with ~144 runtime
threads, slowed the first pooled shard to 47 s and never finished the second within 90 s
(`tools/repro_par_forest.py`, `verify --lanes par-forest`). The AMD MI325X box (no quota) is unaffected.

**Need.** (1) A documented way to set the runtime's worker count per process (an environment variable or an API
called before the first parallel launch), and (2) a default that respects the process's affinity mask AND its cgroup
CPU quota (cgroup v2 `cpu.max`, v1 `cpu.cfs_quota_us`/`cpu.cfs_period_us`), as Go, the JVM and .NET do. The runtime
libraries contain an undocumented `MODULAR_THREAD_BUSY_WAIT_US` (worker busy-wait before sleeping) and a
`runtime.worker_stack_size_mb` key, but no documented worker-count setting
(https://max.modular.com/environment-variables lists none), so we do not set either.

**Where.** `mojolearn._parallel_pool.DevicePool` (the parallel drivers: fit_forest and the other non-cooperative
drivers), the verifier's worker processes, and any user process that runs several mojolearn processes in one
container.

**Meanwhile.** The pool keeps one worker per device, created once and reused for every request; the parent blocks
in a pipe read (no spin) while workers fit; each worker's host loops get `MOJOLEARN_CPU_THREADS` = the effective CPU
budget (min of affinity, cgroup quota, an explicit MOJOLEARN_CPU_THREADS) divided by the worker count
(`_parallel_pool.effective_cpu_budget` / `worker_cpu_threads`). That bounds our own host splits; the Mojo runtime's
own worker count stays the runtime's until Modular ships the knob.

## Host shared-library bytes depend on the build machine's CPU despite an explicit `--target-cpu x86-64-v3` (2026-10-31, release-host-owner)

**What we see.** Release 0.8.63 (source 6b2e4bc320, mojo-1.0.0 / max-core-26.5.0, the same pixi lock and the same
container image on every leg) compiles each merged CPU host shared library with `mojo build --target-cpu x86-64-v3`
(`bindings/merged_build.sh host --group <stem> --rootdir /tmp/mojolearn-merged-roots/host`) on three GitHub
ubuntu-24.04 runners. The recorded inputs (source closure digest, compiler bytes, flags, environment except a
CUDA-only staging variable no host command reads) are identical on all three. Four of the five host libraries come
out byte-identical. `_mojolearn_native_neural_host.so` does not: the runners on AMD EPYC 9V45 and EPYC 9V74 produce
848fbdb544dd..., the runner on AMD EPYC 7763 produces b02d72c83e8e... The files are the same size and have the same
strings. 27,354 bytes differ. The bytes follow the machine that compiled them, not the requested target.

**Need.** For an explicit `--target-cpu`, the host code (and any cost model, tuning or feature detection behind it)
should depend only on the requested target and the inputs, never on the build machine's CPU (its `cpuid`,
microarchitecture or core count). If some pass reads host properties, document it and add a flag to turn it off.
Byte-reproducible builds across machines are how we prove a release's host bindings are the same as the ones we
tested.

**Where.** Every Linux release build: the host bindings ship once in `mojolearn-core`, and the release proves that
every build leg produced them.

**Meanwhile.** `tools/release.py` linux-wait passes a host byte mismatch with identical recorded inputs to
`tools/release_host_composition.py`, which ships the host owner leg's (cuda/sm_89) retained bytes in every set under
a witnessed receipt. Any difference in source, compiler or flags is still refused. Evidence:
`~/mojolearn-evidence/release/0.8.63/6b2e4bc3208d/legs/<leg>/GHA/shards/6/box.txt` (runner CPU) and
`.../release-build/build/compile-inputs.json` (recorded inputs).
