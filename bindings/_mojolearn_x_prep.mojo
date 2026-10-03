# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S GPU BINDING (preprocessing additions, naive Bayes and
discriminant analysis). One entry runs a program of units on the device
(x_prep/common.mojo); the host binding runs the same units on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from x_prep.device import run_program_device, run_program_device_ranges, x_prep_ctx, X_PREP_STORE
from x_prep.folds import I32P, kfold_folds, strat_folds
from x_prep.calib import CALIB_FOLDS, CAL_ST, CAL_LS
from x_prep.fastpt import PTIMPUTE_FLAGS
from x_prep.fastnb_csr import NB_TEXT_CSR, nb_csr_fit_py, nb_csr_jll_py


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s)
    return PythonObject(s)


def run_scratch_binding(arena_addr: PythonObject, arena_len: PythonObject, scratch_len: PythonObject,
                        prog_addr: PythonObject, stages: PythonObject) raises -> PythonObject:
    """x_prep_run with scratch_len device-only words after the arena (lane prep-apple2)."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var sc = Int(py=scratch_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, prog_addr: PythonObject, out_addr: PythonObject,
                    sizes: PythonObject) raises -> PythonObject:
    """x_prep_run_scratch plus one OUTPUT region after the scratch, zeroed on
    the device and copied back into the host buffer at out_addr (lane
    prep-apple2). sizes = (arena_len, scratch_len, out_len, stages)."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=sizes[0])
    var sc = Int(py=sizes[1])
    var on = Int(py=sizes[2])
    var s = Int(py=sizes[3])
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or on < 0 or s < 0 or (on > 0 and oa == 0):
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc, oa, on)
    return PythonObject(s)


def run_ranges_binding(arena_addr: PythonObject, prog_addr: PythonObject, out_addr: PythonObject,
                       sizes: PythonObject, ranges: PythonObject) raises -> PythonObject:
    """x_prep_run_out whose host arena crosses by ranges (lane py-shared,
    core/arena_io.mojo). sizes = (arena_len, scratch_len, out_len, stages);
    ranges = (ins_addr, nins, outs_addr, nouts): Int32 triples [lo, hi, src]
    and quads [lo, hi, CNT, mult] inside the host arena."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=sizes[0])
    var sc = Int(py=sizes[1])
    var on = Int(py=sizes[2])
    var s = Int(py=sizes[3])
    var ia = Int(py=ranges[0])
    var ni = Int(py=ranges[1])
    var ra = Int(py=ranges[2])
    var no = Int(py=ranges[3])
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or on < 0 or s < 0 or (on > 0 and oa == 0) or ni < 0 or no < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device_ranges(fa, n, qa, s, sc, oa, on, ia, ni, ra, no)
    return PythonObject(s)


def dev_put_binding(addr: PythonObject, n_words: PythonObject) raises -> PythonObject:
    """A resident copy of n_words host words (core/device_store.mojo); its id."""
    var a = Int(py=addr)
    var n = Int(py=n_words)
    var id: Int
    with GILReleased(Python()):
        id = X_PREP_STORE.get_or_create_ptr()[].put(x_prep_ctx(), a, n)
    return PythonObject(id)


def dev_free_binding(id: PythonObject) raises -> PythonObject:
    var i = Int(py=id)
    with GILReleased(Python()):
        X_PREP_STORE.get_or_create_ptr()[].free(x_prep_ctx(), i)
    return PythonObject(None)


def dev_live_binding() raises -> PythonObject:
    return PythonObject(X_PREP_STORE.get_or_create_ptr()[].live)


def _seed(v: PythonObject) raises -> UInt64:
    """(lo, hi) 32-bit halves -> the 64-bit seed."""
    return (UInt64(Int(py=v[1])) << 32) | UInt64(Int(py=v[0]))


def strat_folds_binding(codes_addr: PythonObject, out_addr: PythonObject, ints: PythonObject,
                        seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's stratified fold assignment on the host (x_prep/folds.mojo).
    ints = (n, n_classes, n_folds, shuffle); seed = (lo, hi). Returns 0, or -1
    when every class has fewer rows than n_folds."""
    var ca = Int(py=codes_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if ca == 0 or oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    var r = strat_folds(I32P(unsafe_from_address=ca), n, Int(py=ints[1]), Int(py=ints[2]), _seed(seed),
                        Int(py=ints[3]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(r)


def kfold_folds_binding(out_addr: PythonObject, ints: PythonObject, seed: PythonObject) raises -> PythonObject:
    """TargetEncoder's K-fold assignment on the host (x_prep/folds.mojo).
    ints = (n, n_folds, shuffle); seed = (lo, hi)."""
    var oa = Int(py=out_addr)
    var n = Int(py=ints[0])
    if oa == 0 or n < 0:
        raise Error("x_prep: invalid fold buffers")
    kfold_folds(n, Int(py=ints[1]), _seed(seed), Int(py=ints[2]) != 0, I32P(unsafe_from_address=oa))
    return PythonObject(0)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


#: lane/apple-fast-prep (2026-10-02): the encoders' category fit takes the
#: chunked per-column run scan (uniq_count / uniq_scan / uniq_write, existing
#: units) in place of the one-thread-per-column `unique_cols` on the FAST
#: tier on Apple. DEFAULT since the M3 A/B (lane/apple-fast-prep 387211293,
#: n=1, output digests identical: onehot taxi 69.7 -> 30.4 ms, ordinal taxi
#: 67.2 -> 25.3 ms). It was the env switch MOJOLEARN_X_PREP_FAST_UNIQUE=1;
#: now `x_prep_fast_unique` is registered only when this is on, the Python
#: side probes it once per binding (no env read), and
#: `-D MOJOLEARN_X_PREP_FAST_UNIQUE_OFF` restores main's path. The old env
#: name stays harmless.
comptime X_PREP_FAST_UNIQUE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_X_PREP_FAST_UNIQUE_OFF"]()
)


def fast_unique_binding() raises -> PythonObject:
    """Lane apple-fast-prep: present only under X_PREP_FAST_UNIQUE."""
    return PythonObject(1)


def ptimpute_flags_binding() raises -> PythonObject:
    """Lane af-ptimpute (FAST + Apple, each switch its own define, default off):
    the bits of x_prep/fastpt.mojo PTIMPUTE_FLAGS (1 PT_COLBATCH, 2 PT_SPEC,
    4 PT_FUSED_TRANSFORM, 8 SI_ONEPASS, 16 PT_FOLD_NOX); registered only
    when one is on, so the Python layer's probe is the switch."""
    return PythonObject(PTIMPUTE_FLAGS)


def calib_folds_binding() raises -> PythonObject:
    """Lane apple-fast-meta (FAST + Apple default, -D MOJOLEARN_CALIB_GNB_FOLDS_OFF turns it off):
    the CalibratedClassifierCV(GaussianNB) program's constants [words per
    Platt problem, line-search steps]; registered only when the ops exist."""
    var out = Python.list()
    out.append(PythonObject(CAL_ST))
    out.append(PythonObject(CAL_LS))
    return out



@export
def PyInit__mojolearn_x_prep() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep")
        m.def_function[run_binding]("x_prep_run")
        comptime if NB_TEXT_CSR:
            # lane apple-fast-nb: FAST + Apple + -D MOJOLEARN_NB_TEXT_CSR only (x_prep/fastnb_csr.mojo)
            m.def_function[nb_csr_fit_py]("x_prep_nb_csr_fit")
            m.def_function[nb_csr_jll_py]("x_prep_nb_csr_jll")
        m.def_function[run_scratch_binding]("x_prep_run_scratch")
        m.def_function[run_out_binding]("x_prep_run_out")
        m.def_function[run_ranges_binding]("x_prep_run_ranges")
        m.def_function[dev_put_binding]("x_prep_dev_put")
        m.def_function[dev_free_binding]("x_prep_dev_free")
        m.def_function[dev_live_binding]("x_prep_dev_live")
        m.def_function[strat_folds_binding]("x_prep_strat_folds")
        m.def_function[kfold_folds_binding]("x_prep_kfold_folds")
        m.def_function[numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[vendor_binding]("x_prep_vendor")
        comptime if X_PREP_FAST_UNIQUE:
            m.def_function[fast_unique_binding]("x_prep_fast_unique")
        comptime if CALIB_FOLDS:
            m.def_function[calib_folds_binding]("x_prep_calib_folds")
        comptime if PTIMPUTE_FLAGS != 0:
            m.def_function[ptimpute_flags_binding]("x_prep_ptimpute_flags")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep: ", e))
