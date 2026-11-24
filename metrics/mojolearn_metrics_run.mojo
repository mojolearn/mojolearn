# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN pieces of `bindings/_mojolearn_metrics.mojo`: the typed address casts
and loads, the spectral output guards, the output copies of the spectral
doors, and the typed bodies of `precision_recall_curve`, `confusion_matrix`
and `spectral_predict`. Typed Mojo only, no Python import
(tools/python_free_dirs.sh). The door keeps the argument reads and releases
the GIL around each body exactly where it did before; the bodies are moved
unchanged, so the bits are unchanged (lane rs-metrics-estimators,
2026-11-12)."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from metrics.estimator import binary_ranking_host, confusion_matrix_host


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _load_i32(addr: Int, n: Int) raises -> List[Int32]:
    return read_i32(addr, max(0, n))


def _load_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, max(0, n))


def precision_recall_curve_run(
    y: List[Int32], scores: List[Float32], n: Int,
    precision: MutPointer[Float32, MutUntrackedOrigin], recall: MutPointer[Float32, MutUntrackedOrigin], thresholds: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    # Called with the GIL released by the door (precision_recall_curve_binding).
    var m = 0
    var result = binary_ranking_host[True](y,scores,n)
    m = result[1]
    var curve = result[0].unsafe_ptr()
    unsafe_memcpy(dest=precision, src=curve, count=m + 1)
    unsafe_memcpy(dest=recall, src=curve + (n + 1), count=m + 1)
    unsafe_memcpy(dest=thresholds, src=curve + 2 * (n + 1), count=m)
    _ = len(result[0])
    return m


def confusion_matrix_run(
    y: List[Int32], p: List[Int32], n: Int, k: Int, normalization: Int, address: Int,
) raises:
    # Called with the GIL released by the door (confusion_matrix_binding);
    # normalization 0 writes Int64 counts, otherwise Float32.
    if normalization == 0:
        var output = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)
        var values = confusion_matrix_host[DType.int64](y,p,n,k,normalization)
        unsafe_memcpy(dest=output, src=values.unsafe_ptr(), count=len(values))
        _ = len(values)
    else:
        var output = _f32_ptr(address)
        var values = confusion_matrix_host[DType.float32](y,p,n,k,normalization)
        unsafe_memcpy(dest=output, src=values.unsafe_ptr(), count=len(values))
        _ = len(values)


# TOMBSTONE: the spectral door helpers (_guard_spectral_outputs, _spectral_write_outputs, _guard_embedding_output,
# _spectral_write_embedding, _spectral_state_copy, spectral_predict_run) of the deleted by-name spectral exports
# deleted 2026-11-23 on lane/sab-host-missed-cls (no other caller); code recoverable at 4519097d38.
