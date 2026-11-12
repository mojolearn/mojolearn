# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN piece of `providers/python/metrics_vector_input_pyapi.mojo`: the
int32 narrowing of an admitted integer label vector and its range refusal.
Typed Mojo only, no Python import (tools/python_free_dirs.sh); the call and
the refusal text are unchanged (lane rs-metrics-estimators, 2026-11-12)."""
from core.admitted_ids import admitted_ids_i32


def _metric_i32_range(name: String) raises:
    raise Error("mojolearn metrics: " + name + " does not fit in int32; the implemented "
                + "kernels are the int32 instantiation (the int64 overload of "
                + "adjusted_rand_index is the same code at a wider type and is "
                + "not instantiated, metrics/NOT_IMPLEMENTED.tsv)")


def metric_i32_narrow(
    addr: Int, stride: Int, count: Int, itemsize: Int, signed: Bool, dst_addr: Int, name: String,
) raises:
    # admit_ids: one parallel read narrowed to int32 in the same pass.
    if not admitted_ids_i32(addr, stride, 0, count, 1, itemsize, signed, False,
                            MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_addr)):
        _metric_i32_range(name)
