# SPDX-License-Identifier: Apache-2.0
# Native unit threads (lane p3-msel-parallel-and-process-pool, 2026-11-09).
#
# Independent units that call compiled owners through CPython objects (a
# GridSearchCV candidate x fold fit, a cross-validation fold) run on several
# OS threads of THIS process: no child interpreter, no package import, no
# pickle. The caller runs units too; `workers - 1` more threads are started
# with `pthread_create` and joined with the GIL released.
#
# Why OS threads and not `host_parallelize` (core/host_parallel.mojo): a unit
# holds the GIL while it drives its owner and gives it up only inside the
# owners' GIL-released kernel calls. A runtime pool worker blocked on the GIL
# could be the one a nested `host_parallelize` inside another unit's fit
# waits for: a deadlock. Threads of our own never serve the runtime pool.
#
# The entry function owns the unit protocol: it takes the GIL with
# `native_unit_gil_enter` (reentrant: the caller already holds it), drains
# units in index order under the GIL, writes each result into the slot of its
# unit (the order of the results is the unit order, never the finish order)
# and gives the GIL back with `native_unit_gil_leave`. The floating-point
# environment of the caller is installed on every worker (core/host_parallel
# DEVIATION 5900), so a unit computes the same bits on any thread.
from std.atomic import Atomic, Ordering
from std.ffi import c_int, c_size_t, external_call
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import num_physical_cores

from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.optimizer_public_words import stash_python_error, PYTHON_ERROR_PENDING, _restore_python_error

#: The thread entry: `void *(*)(void *)` with the argument and result as
#: address words (same register class under the C ABI on every target).
comptime NativeUnitEntry = def(Int) thin abi("C") -> Int

#: Words reserved for a `pthread_attr_t` (56 bytes on glibc x86_64/aarch64,
#: 64 bytes on macOS): 128 bytes.
comptime _ATTR_WORDS = 16
#: Stack of every unit thread: owners recurse over model structure (nested
#: estimators, tree builders); a macOS secondary thread gets 512 KiB by
#: default, the main thread 8 MiB. Reserved address space, not resident.
comptime _UNIT_THREAD_STACK_BYTES = 64 << 20


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


def native_unit_cores() -> Int:
    """Physical cores of this host (at least 1)."""
    return max(num_physical_cores(), 1)


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


def native_unit_gate() -> List[Int64]:
    """A gate one unit holds at a time (a unit's gathers or scores that share
    an owner's program state): one zeroed word on the heap."""
    return List[Int64](length=1, fill=0)


def native_unit_gate_enter(mut gate: List[Int64]):
    """Take the gate, yielding with the GIL released while another unit
    holds it. The word is read and written atomically: a plain Bool field of
    the run, read through `mut self`, may be loaded once and never again
    (the compiler may assume no other thread writes it), so a waiter spun
    forever on a gate its holder had already opened (cross_val_score on 2+
    threads, 2026-11-09)."""
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](gate.unsafe_ptr())
    while True:  # driver-loop: wait for another unit's gated section (GIL released while waiting)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return
        with GILReleased(Python()):
            _ = external_call["sched_yield", c_int]()


def native_unit_gate_leave(mut gate: List[Int64]):
    """Open the gate `native_unit_gate_enter` took."""
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](gate.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def native_unit_gil_enter() raises -> c_int:
    """`PyGILState_Ensure` from the one CPython table (create the table on
    the caller thread before starting workers)."""
    var table = cpython_table()
    return table[].PyGILState_Ensure()


def native_unit_gil_leave(state: c_int) raises:
    var table = cpython_table()
    table[].PyGILState_Release(state)


def native_unit_threads_run(entry: NativeUnitEntry, arg: Int, workers: Int) raises:
    """Run `entry(arg)` on the calling thread and on `workers - 1` new OS
    threads, then join them with the GIL released. The caller must hold the
    GIL. A thread that cannot be started is not an error: the units it would
    have taken are drained by the others (the entry pulls units, it is not
    handed a fixed share)."""
    # Created here, with the GIL held, so no worker creates it.
    _ = cpython_table()
    var extra = workers - 1
    var handles = List[Int](length=max(extra, 0), fill=0)
    var attr = List[Int64](length=_ATTR_WORDS, fill=Int64(0))
    var started = 0
    if extra > 0 and external_call["pthread_attr_init", c_int](Int(attr.unsafe_ptr())) == 0:
        _ = external_call["pthread_attr_setstacksize", c_int](Int(attr.unsafe_ptr()), c_size_t(_UNIT_THREAD_STACK_BYTES))
        for i in range(extra):  # driver-loop: over unit threads (at most one per core)
            var slot = Int(handles.unsafe_ptr()) + i * 8
            if external_call["pthread_create", c_int](slot, Int(attr.unsafe_ptr()), entry, arg) != 0:
                break
            started += 1
        _ = external_call["pthread_attr_destroy", c_int](Int(attr.unsafe_ptr()))
    _ = entry(arg)
    if started > 0:
        # Workers still running need the GIL to finish their units.
        with GILReleased(Python()):
            for i in range(started):  # driver-loop: over started unit threads
                _ = external_call["pthread_join", c_int](handles[i], Int(0))
    _ = handles^
    _ = attr^


struct NativeUnitFailure(Movable):
    """The error of the lowest failing unit of a run: the one the serial
    order raises (units are pulled in index order, so every lower unit was
    started before it and ran to its end). A stashed Python exception
    (`PYTHON_ERROR_PENDING`) is fetched out of the process-wide pending slot
    into this record at once, with the GIL held, so another unit's stash
    cannot overwrite it; `reraise` puts it back on the calling thread."""
    var unit: Int
    var message: String
    var words: List[Int]

    def __init__(out self):
        self.unit = -1
        self.message = String()
        self.words = List[Int](length=3, fill=0)

    def failed(self) -> Bool:
        return self.unit >= 0

    def record(mut self, u: Int, error: Error):
        try:
            var words = List[Int](length=3, fill=0)
            var message = String(error)
            if message == PYTHON_ERROR_PENDING:
                _ = _restore_python_error()
                ref cpy = Python().cpython()
                var p = words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
                cpy.lib.get_function[NoneType]("PyErr_Fetch")(p, p + 1, p + 2)
            if self.unit >= 0 and self.unit < u:
                _native_unit_drop(words)
                return
            _native_unit_drop(self.words)
            self.unit = u
            self.message = message
            self.words = words^
        except:
            pass

    def reraise(mut self) raises:
        if self.unit < 0:
            return
        if self.words[0] != 0:
            var w = self.words.copy()
            self.words = List[Int](length=3, fill=0)
            ref cpy = Python().cpython()
            cpy.lib.get_function[NoneType]("PyErr_Restore")(w[0], w[1], w[2])
            var err = stash_python_error()
            raise err^
        raise Error(self.message)


def _native_unit_drop(words: List[Int]):
    """Release a fetched (type, value, traceback) triple."""
    if words[0] == 0 and words[1] == 0 and words[2] == 0:
        return
    try:
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_Restore")(words[0], words[1], words[2])
        cpython_table()[].PyErr_Clear()
    except:
        pass
