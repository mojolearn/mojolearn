# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_svm_host.mojo` (wave 8 run split,
2026-11-12): the IsolationForest host run's output copies and info words.
The door reads its Python addresses and parameters into ints and floats,
keeps the resident take and put under the GIL and the run with the GIL
released exactly as before, then calls one of these. Typed Mojo only, no
Python import (tools/python_free_dirs.sh). Moved word for word, so the bits
are unchanged."""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr
from isolation_forest.estimator import IF_WANT_PREDICT
from svm.host.svm_host_impl import IForestHostRun


def iforest_host_resident_outputs_run(
    run: IForestHostRun,
    want: Int,
    out_i32_address: Int,
    out_f32_address: Int,
    n_query: Int,
    n_features: Int,
    held_token: Int,
    ip: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """The resident run's output (labels for predict, else values) and its
    four info words (slot 3 the forest's token)."""
    if want == IF_WANT_PREDICT:
        unsafe_memcpy(dest=i32_ptr(out_i32_address), src=run.labels.unsafe_ptr(), count=n_query)
    else:
        unsafe_memcpy(dest=f32_ptr(out_f32_address), src=run.values.unsafe_ptr(), count=n_query)
    ip[0] = run.offset_
    ip[1] = Float64(run.max_samples_)
    ip[2] = Float64(n_features)
    ip[3] = Float64(held_token)


def iforest_host_outputs_run(
    run: IForestHostRun,
    want: Int,
    out_i32_address: Int,
    out_f32_address: Int,
    n_query: Int,
    n_features: Int,
    ip: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """The refit-per-call run's output and its three info words."""
    if want == IF_WANT_PREDICT:
        var oi = i32_ptr(out_i32_address)
        unsafe_memcpy(dest=oi, src=run.labels.unsafe_ptr(), count=n_query)
    else:
        var of = f32_ptr(out_f32_address)
        unsafe_memcpy(dest=of, src=run.values.unsafe_ptr(), count=n_query)
    ip[0] = run.offset_
    ip[1] = Float64(run.max_samples_)
    ip[2] = Float64(n_features)
