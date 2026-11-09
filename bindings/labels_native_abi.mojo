# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# The label helpers' native device-or-host selector (lane
# s1-shared-state-hardening-and-selector, S3 = A7c item 4; design note
# ~/mojolearn-evidence/design-shared-process-state.md section 3b).
#
# The label path (bindings/classical_labels.mojo) is compiled into many
# family bindings, host and GPU. Its base-binding steps (prepare_targets,
# reduce_stat, argmax_rows, gather, class_ratio) run on the device on a GPU
# install and on the host on a CPU-only one. They used to reach the base
# binding by name through Python (`_buffer._native` callables in a context
# table, one PyObject_CallObject per step). Now:
#
#   * the GPU base binding (bindings/_mojolearn.mojo) exports
#     `labels_native_api_v1`, the bindings/kit_call.mojo descriptor ABI
#     ("mojolearn.labels.v1") with the operations below over its own device
#     helpers;
#   * the label path connects it ONCE per process (kit_connect) and keeps the
#     provider in the named slot `MojolearnLabelsProviderV1`
#     (bindings/classical_labels.mojo `_labels_provider`); each step is one
#     `kit_call`;
#   * the choice is one test in Mojo: a connected provider runs the step
#     there, an install with no GPU base binding (the import glue passes
#     `labels_base_binding = None`) runs the same host body in-process
#     (bindings/labels_host_ops.mojo, bindings/host_helpers.mojo,
#     bindings/hotpath_helpers.mojo, bindings/array_helpers.mojo: the bodies
#     the core host binding exports under the same names). No py_call, no
#     per-call namespace lookup.
#
# Word layouts (addresses / ints; no floats):
#   PREPARE_TARGETS  [y, classes, codes, w] / [y_code, n, w_code, w_flags] -> k or a PT_W_* status
#   REDUCE_STAT      [src, out(3 words)] / [code, n, what] -> LABELS_RS_* tag, value in out
#   ARGMAX_F32/F64   [scores, dst] / [rows, cols] -> 0
#   GATHER_I32/I64/F64 [table, codes, dst] / [n_table, n] -> 0
#   CLASS_RATIO_F64  [counts, out] / [k, n, mode] -> zero-count flag
# Numerics: none here; the provider runs the base binding's own entries.
from std.memory import bitcast
from std.python import Python, PythonObject

from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type

comptime LABELS_NATIVE_ABI_NAME = "mojolearn.labels.v1"
comptime LABELS_NATIVE_FACTORY = "labels_native_api_v1"

comptime LABELS_OP_MODE = 0
comptime LABELS_OP_CAPABILITIES = 1
comptime LABELS_OP_PREPARE_TARGETS = 2
comptime LABELS_OP_REDUCE_STAT = 3
comptime LABELS_OP_ARGMAX_F32 = 4
comptime LABELS_OP_ARGMAX_F64 = 5
comptime LABELS_OP_GATHER_I32 = 6
comptime LABELS_OP_GATHER_I64 = 7
comptime LABELS_OP_GATHER_F64 = 8
comptime LABELS_OP_CLASS_RATIO_F64 = 9
comptime LABELS_NATIVE_OPERATION_COUNT = 10

#: the capability word of the device provider
comptime LABELS_CAP_DEVICE = 1

#: `reduce_stat`'s answer kinds through the int-only result word
comptime LABELS_RS_INT = 0
comptime LABELS_RS_FLOAT = 1
comptime LABELS_RS_TRIPLE = 2


def labels_reduce_stat_encode(value: PythonObject, out_addr: Int) raises -> Int:
    """Provider side: `reduce_stat`'s Python answer (an int, a float, or the
    exact integer sum's (hi, mid, low)) as words at `out_addr` (3 Int64
    slots) and its tag."""
    if out_addr == 0:
        raise Error("labels native ABI: reduce_stat needs an output span")
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_addr)
    var b = native_loaded_module("builtins")
    var t = py_type(value)
    if t is b.tuple:
        out[0] = Int64(Int(py=value[0]))
        out[1] = Int64(Int(py=value[1]))
        out[2] = Int64(Int(py=value[2]))
        return LABELS_RS_TRIPLE
    if t is b.float:
        out[0] = bitcast[DType.int64](Float64(py=value))
        return LABELS_RS_FLOAT
    out[0] = Int64(Int(py=value))
    return LABELS_RS_INT


def labels_reduce_stat_decode(tag: Int, out_addr: Int) raises -> PythonObject:
    """Client side: the Python answer `reduce_stat` returned, rebuilt from
    the provider's words (the same int, float bits or triple)."""
    var out = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=out_addr)
    if tag == LABELS_RS_TRIPLE:
        return Python.tuple(PythonObject(Int(out[0])), PythonObject(Int(out[1])), PythonObject(Int(out[2])))
    if tag == LABELS_RS_FLOAT:
        return PythonObject(bitcast[DType.float64](out[0]))
    if tag == LABELS_RS_INT:
        return PythonObject(Int(out[0]))
    raise Error("labels native ABI: unknown reduce_stat answer kind " + String(tag))
