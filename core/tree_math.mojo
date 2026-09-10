# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Shared numeric seams for RF and Extra Trees objective kernels.

Source provenance: cuML 00094f7 (ET) and v26.08 265b9da6 (RF),
cpp/src/decisiontree/batched-levelalgo/
objectives.cuh EntropyObjectiveFunction::GainPerSplit calls raft::log.
MojoLearn's existing RF DEVIATION 406 and ET entropy seam wrap that call
through checks.numerics.identical_log for Float32 and std.math.log for other
floating widths. This module extracts their token-identical wrapper; it does
not change arithmetic, dtype dispatch, fusion, or any learner's split policy.
FAST/DETERMINISTIC retain stdlib Float32 log; IDENTICAL retains portable_logf
through the existing identical_log function. This is a host/device scalar
primitive, not a CPU training or prediction implementation.
"""
from std.math import log
from checks.numerics import identical_log


@always_inline
def tree_log[
    dt: DType, //
](x: Scalar[dt]) -> Scalar[dt] where dt.is_floating_point():
    """The shared RF/ET generic logarithm seam, with existing mode dispatch."""
    comptime if dt == DType.float32:
        return identical_log(x.cast[DType.float32]()).cast[dt]()
    return log(x)
