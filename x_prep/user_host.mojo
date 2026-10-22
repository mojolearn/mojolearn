# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The typed pointer aliases the host x_prep binding reads its buffers through.
The IterativeImputer(estimator=...) host walks that lived here run on the
device now (x_prep/iterative.mojo)."""

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime F64P = MutPointer[Float64, MutAnyOrigin]

