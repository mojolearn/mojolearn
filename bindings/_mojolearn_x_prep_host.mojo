"""CPU binding for `_mojolearn_x_prep`. HOST ONLY: the same units as the
device, run in a loop on the caller's arena (x_prep/host/program.mojo), with
the GPU binding's export names and address contract."""
from core.experiment_guards_base import BASE_GUARDS_OK
from core.classical_identical_controls.shared_controls import C08_DICTIONARY, C08_TARGET_CODES
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from core.shared_slot import SharedSlot
from core.scaler_native_api import PrepProgramAPI
from x_prep.transport_host import publish_x_prep_transport_host
from core.prep_program_native_host import PrepProgramHostKernels
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.common import X_PREP_HOST_SABOTAGE
from x_prep.host.program import run_program_host
from x_prep.user_host import F32P, F64P, I32P
from x_prep.label_fast import IDN_LABEL
from x_prep.blocked import IDN_NB_ONEPASS
from x_prep.blocked import IDN_STATS_BLOCKED, IDN_CLASS_ONEPASS
from x_prep.select_blocked import IDN_SELECT_BLOCKED
from x_prep.pt_blocked import IDN_PT_BLOCKED
from x_prep.host.rr_eigh_host import IDN_RR_EIGH
from x_prep.fam2 import IDN_WDRAW, IDN_PERM_DRAW, IDN_WPICK, IDN_PARTIAL_CODES, IDN_LABEL_INV
from x_prep.gram_blocked import IDN_GRAM_BLOCKED, IDN_GRAM_ROWS
from x_prep.proba64 import PROBA64


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def x_prep_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_prep_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_prep_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_prep host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_prep_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_PREP_HOST_SABOTAGE)


def x_prep_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_prep_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def _seed(v: PythonObject) raises -> UInt64:
    """(lo, hi) 32-bit halves -> the 64-bit seed."""
    return (UInt64(Int(py=v[1])) << 32) | UInt64(Int(py=v[0]))


def proba64_binding() raises -> PythonObject:
    """Lane apple-fast-q-clf (x_prep/proba64.mojo): registered only under
    PROBA64 (FAST, not -D MOJOLEARN_PROBA64_QOLD); Python's probe for staging
    `q64_softmax` (float64 predict_proba)."""
    return PythonObject(1)


# TOMBSTONE: x_prep_calib_folds, x_prep_maxabs_pool, x_prep_py2mojo (dead native entries: constant / reach probes nothing reads) deleted 2026-10-26 by bb4259d55; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_prep.patch


def label_present_binding() raises -> PythonObject:
    return PythonObject(1)


def idn_int_binding() raises -> PythonObject:
    """Lane idn-int-prep: bindings/_mojolearn_x_prep.mojo `idn_int_binding`'s
    bits for the host column (1 IDN_LABEL, 2 IDN_NB_ONEPASS; the CSR entry
    is the device's only)."""
    return PythonObject((1 if IDN_LABEL else 0) | (2 if IDN_NB_ONEPASS else 0))


def c08_routes_binding() raises -> PythonObject:
    """C08 routes compiled in (lane classical-encoders): 1 TargetEncoder fit codes,
    2 retired (C08 OneHotEncoder arm, deleted 2026-11-15), 4 retired (C08 OrdinalEncoder arm, deleted 2026-11-15)."""
    # TOMBSTONE: MOJOLEARN_CLASSICAL_C08_ORDINAL_FT (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 46115a2bb3.
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C08_ORDINAL_FT.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    # TOMBSTONE: MOJOLEARN_CLASSICAL_C08_ONEHOT_FT (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 59c24bdf40.
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C08_ONEHOT_FT.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    return PythonObject(Int(C08_TARGET_CODES))


def classical_shared_binding() raises -> PythonObject:
    """Bits read by python/mojolearn/_expansion_prep.py `_classical_shared`:
    1 C08_DICTIONARY, 2 retired (C55, deleted 2026-10-07), 4 retired (C04 LDA, deleted 2026-11-15),
    8 retired (C56 LDA input, deleted 2026-11-15); 16-48 and 64 retired (C61 NB arms / DA, deleted 2026-10-08
    after grid ge123e6f9: slower on both vendors, quality same)."""
    # TOMBSTONE: MOJOLEARN_CLASSICAL_C56_LDA_INPUT (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 859362ebd3.
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C56_LDA_INPUT.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    # TOMBSTONE: MOJOLEARN_CLASSICAL_C04_LDA (opt-in arm, EXP:1532 DROPPED-serial) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 93b7a49f96.
    # Restore: git apply experiments/removed/MOJOLEARN_CLASSICAL_C04_LDA.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    return PythonObject(
        Int(C08_DICTIONARY)
    )


def idn_fam_binding() raises -> PythonObject:
    """Lane fam-prep-metrics (IDENTICAL, device and host column alike): the
    bits of the family switches this binding was built with, read by
    python/mojolearn/_expansion_prep.py `_idn_fam` (1 IDN_STATS_BLOCKED, 2
    IDN_CLASS_ONEPASS: op 165, 4 IDN_SELECT_BLOCKED: ops 166-171, 8
    IDN_PT_BLOCKED: ops 172-176, 16 IDN_RR_EIGH: informational, the eigh
    stage's order is the binding's own); registered only when one is on."""
    return PythonObject(
        (1 if IDN_STATS_BLOCKED else 0) | (2 if IDN_CLASS_ONEPASS else 0) | (4 if IDN_SELECT_BLOCKED else 0)
        | (8 if IDN_PT_BLOCKED else 0) | (16 if IDN_RR_EIGH else 0)
    )


def idn_fam2_binding() raises -> PythonObject:
    """Lane fam2-prep-metrics (IDENTICAL, device and host column alike): the
    bits of the x_prep/fam2.mojo switches this binding was built with, read
    by python/mojolearn/_expansion_prep.py `_idn_fam2` (1 IDN_WDRAW: ops
    230-232, 2 IDN_PERM_DRAW: op 233, 4 IDN_WPICK: op 234, 8
    IDN_PARTIAL_CODES: op 235, 16 IDN_GRAM_BLOCKED: ops 236, 238-240, 32
    retired (IDN_GRAM_ROWTILE, deleted 2026-11-15), 64 IDN_LABEL_INV: op 241, bits 16 and up: the Gram's rows per
    block); registered only when one is on."""
    # TOMBSTONE: MOJOLEARN_IDN_GRAM_ROWTILE (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at ba54f83c9b.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAM_ROWTILE.expdel2-02-base.patch; record in docs/TOMBSTONES.md.
    return PythonObject(
        (1 if IDN_WDRAW else 0) | (2 if IDN_PERM_DRAW else 0) | (4 if IDN_WPICK else 0)
        | (8 if IDN_PARTIAL_CODES else 0) | (16 if IDN_GRAM_BLOCKED else 0)
        | (64 if IDN_LABEL_INV else 0) | (IDN_GRAM_ROWS << 16)
    )


# TOMBSTONE: MOJOLEARN_IDN_LAB_UNIQUE_HIST, MOJOLEARN_IDN_LAB_CODES_DIRECT, MOJOLEARN_IDN_LAB_ONE_ENTRY (losers: label-encoder 2.1-2.6x, label-binarizer istella 1.15-1.39x slower, NV + AMD) deleted 2026-10-11 by de9ce7b39; code recoverable at de9ce7b39^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch
# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_x_prep_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
comptime PREPPROGRAM_HOST_NATIVE_API = SharedSlot[PrepProgramAPI[PrepProgramHostKernels], "MojolearnPrepProgramHostNativeV1"]


def x_prep_native_api_v1_binding() raises -> PythonObject:
    var api = PREPPROGRAM_HOST_NATIVE_API.get_or_create_ptr()
    return neural_descriptor_capsule(api[].name, api[].entries)


def build__mojolearn_x_prep_host() -> PythonObject:
    comptime assert BASE_GUARDS_OK, "invalid integrated experiment configuration (base guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep_host")
        m.def_function[x_prep_native_api_v1_binding]("x_prep_native_api_v1")
        m.def_function[x_prep_host_numeric_mode_binding]("x_prep_host_numeric_mode")
        m.def_function[x_prep_host_vendor_binding]("x_prep_host_vendor")
        m.def_function[x_prep_host_column_binding]("x_prep_host_column")
        m.def_function[x_prep_host_sabotage_binding]("x_prep_host_sabotage")
        m.def_function[run_binding]("x_prep_run")
        m.def_function[x_prep_numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[x_prep_vendor_binding]("x_prep_vendor")
        # TOMBSTONE: x_prep_calib_folds, x_prep_maxabs_pool, x_prep_py2mojo (dead native entries: constant / reach probes nothing reads) deleted 2026-10-26 by bb4259d55; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_x_prep.patch
        # lane idn-int-prep: the host column runs the same programs as the
        # IDENTICAL device binding (x_prep/label_fast.mojo, x_prep/blocked.mojo)
        comptime if IDN_LABEL:
            m.def_function[label_present_binding]("x_prep_label_present")
        comptime if IDN_LABEL or IDN_NB_ONEPASS:
            m.def_function[idn_int_binding]("x_prep_idn_int")
        comptime if IDN_STATS_BLOCKED or IDN_CLASS_ONEPASS or IDN_SELECT_BLOCKED or IDN_PT_BLOCKED or IDN_RR_EIGH:
            # lane fam-prep-metrics
            m.def_function[idn_fam_binding]("x_prep_idn_fam")
            m.def_function[classical_shared_binding]("x_prep_classical_shared")
            m.def_function[c08_routes_binding]("x_prep_c08_routes")
        comptime if IDN_WDRAW or IDN_PERM_DRAW or IDN_WPICK or IDN_PARTIAL_CODES or IDN_GRAM_BLOCKED or IDN_LABEL_INV:
            # lane fam2-prep-metrics: the fam2 switches (x_prep/fam2.mojo)
            m.def_function[idn_fam2_binding]("x_prep_idn_fam2")
        comptime if PROBA64:
            m.def_function[proba64_binding]("x_prep_proba64")
        var published_module = m.finalize()
        # The native program transport (core/prep_entry_table.mojo) for the
        # native owners (bindings/x_prep_native_owner.mojo xprep_run).
        publish_x_prep_transport_host()
        return published_module
    except e:
        abort(String("failed to create _mojolearn_x_prep_host: ", e))
