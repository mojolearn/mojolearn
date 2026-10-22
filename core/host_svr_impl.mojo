# SPDX-License-Identifier: Apache-2.0
# Native scalar/index preparation for the unchanged explicit CPU SVM ABI.
from std.math import isfinite
from core.classical_pandas_impl import pd_raw_at, _pd_double


def host_svr_bias(source: Int, width: Int, kind: Int, native: Bool) -> Float64:
    return _pd_double(pd_raw_at(source, 0, width, native), kind, width)


def host_svr_indices(source: Int, stride: Int, count: Int, width: Int, kind: Int,
                     native: Bool, columns: Int, destination: Int) raises:
    var out = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=destination)
    for i in range(count):
        var value = _pd_double(pd_raw_at(source, i * stride, width, native), kind, width)
        if not isfinite(value) or value < -2147483648.0 or value >= 2147483648.0:
            raise Error("precomputed support indices must fit int32")
        var index = Int(value)
        if index < 0 or index >= columns:
            raise Error("precomputed support index exceeds the query columns")
        out[i] = Int32(index)
