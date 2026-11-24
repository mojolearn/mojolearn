# SPDX-License-Identifier: Apache-2.0
# The unit job bounds of a door (lane p3-msel-parallel-and-process-pool,
# 2026-11-09): `native_unit_jobs` (an estimator's `n_jobs`) and
# `native_unit_device_workers` (one at a time when the members run on a GPU).
#
# REMOVED (lane mf-r9b, 2026-11-24): the native unit threads (OS threads of
# this process running Python units: `native_unit_threads_run`, the GIL
# protocol `native_unit_gil_enter` / `_leave`, `native_unit_gate_enter`
# and `NativeUnitFailure`). Their one user, the Python cv units
# (bindings/msel_cv_door.mojo `msel_cv_units`), is deleted with the search
# door fallback; recoverable at ea2e0c0139.
from std.python import PythonObject
from std.sys.compile import is_defined

from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.native_unit_os_threads_run import native_unit_cores


#: Units whose member fits run on a GPU run ONE AT A TIME (orchestrator
#: decision, integration w6, 2026-11-09). On the Mac (Metal) two units fitting
#: at once crash in Metal command-buffer completion (MTLResourceList
#: releaseAllObjectsAndReset / useResource; cross_val_score and
#: CalibratedClassifierCV with n_jobs >= 2 or None, GaussianNB, LinearSVC,
#: LogisticRegression members): the member fits share the process-wide
#: device context (core/neural_context.mojo `process_ctx`), and the owners'
#: caches are not thread-safe. The device kernels are themselves parallel, so
#: one unit at a time still fills the GPU. Host-only units (a CPU-only install,
#: the host column: context vendor "cpu") keep the parallel units.
#: -D MOJOLEARN_MSEL_UNITS_GPU_PARALLEL keeps the parallel-GPU arm compiled,
#: default off: it awaits a thread-safety lane (per-thread contexts, cache
#: audit) and then an NVIDIA + AMD A/B (RUN OWED: GridSearchCV /
#: cross_validate / CalibratedClassifierCV rows of paper_compare, arm on vs off).
comptime NATIVE_UNITS_GPU_PARALLEL = is_defined["MOJOLEARN_MSEL_UNITS_GPU_PARALLEL"]()
# TOMBSTONE: MOJOLEARN_MSEL_UNITS_GPU_PARALLEL (opt-in arm, crashes on Metal, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 853bb6d0ee.
# Restore: git apply experiments/removed/MOJOLEARN_MSEL_UNITS_GPU_PARALLEL.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/restore-apple-fast-classical: Apple fast is paused, not abandoned (Andrew); opt-in, default off, Apple fast only.


def native_unit_device_workers(workers: Int, context: PythonObject) -> Int:
    """`workers`, or 1 when the units' member fits run on a GPU: the vendor the
    process resolved (context "vendor": "metal", "cuda", "hip"; "cpu" on a
    CPU-only install or the host column). An unknown vendor counts as a GPU."""
    comptime if NATIVE_UNITS_GPU_PARALLEL:
        return workers
    try:
        var vendor = py_get(context, PythonObject("vendor"), PythonObject(None))
        if vendor is not None and String(py=vendor) == "cpu":
            return workers
    except:
        pass
    return min(workers, 1)


def native_unit_jobs(n_jobs: PythonObject, caller: String) raises -> Int:
    """The thread bound of a unit run from an estimator's `n_jobs`
    (joblib's meaning): an int k > 0 is the bound, -1 all cores, -k all
    cores but k - 1; None all cores (parallel by default, AGENTS.md: the
    results do not depend on it)."""
    var cores = native_unit_cores()
    if n_jobs is None:
        return cores
    if py_type(n_jobs) is not native_loaded_module("builtins").int:
        raise Error("TypeError: " + caller + ": n_jobs must be None or an int")
    var k = Int(py=n_jobs)
    if k == 0:
        raise Error("ValueError: n_jobs == 0 in Parallel has no meaning")
    if k < 0:
        return max(cores + 1 + k, 1)
    return k
