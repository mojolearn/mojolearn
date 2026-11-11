# SPDX-License-Identifier: Apache-2.0
"""Spectral RBF affinity's existing cell sequence with native scratch/handles."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_decomp.descriptor_schema import (
    XD_EW, XD_SQDIST, XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD,
    XD_DEV_DOWNLOAD, XD_DEV_EW, XD_DEV_SQDIST,
)


@fieldwise_init
struct SpectralRBFRequest(ImplicitlyCopyable, Movable):
    var source: Int
    var destination: Int
    var rows: Int
    var features: Int
    var gamma: Float64
    var resident: Bool


struct SpectralRBFCells(Movable):
    var entry: NativeDescriptorCall
    var handles: List[Int]

    def __init__(out self, entry: NativeDescriptorCall):
        self.entry = entry
        self.handles = List[Int]()

    def allocate(mut self, count: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 1] = [count]
        var f: Array[Float64, 0] = []
        var handle = native_descriptor_call_fixed(self.entry, XD_DEV_ALLOC, a, p, f)
        self.handles.append(handle)
        return handle

    def __del__(deinit self):
        # Preserve the former ascending allocation-order quiet pool return.
        for i in range(len(self.handles)):  # driver-loop: at most five RBF operation buffers
            try:
                var a: Array[Int, 0] = []
                var p: Array[Int, 1] = [self.handles[i]]
                var f: Array[Float64, 0] = []
                _ = native_descriptor_call_fixed(self.entry, XD_DEV_FREE, a, p, f)
            except:
                pass


def spectral_rbf_affinity(entry: NativeDescriptorCall, request: SpectralRBFRequest) raises:
    var n = request.rows
    var d = request.features
    var nn = n * n
    var scale: Array[Float64, 1] = [-request.gamma]
    var scalar_zero: Array[Float64, 1] = [0.0]
    var distance: Array[Float64, 1] = [2.0]
    var no_floats: Array[Float64, 0] = []
    var no_addresses: Array[Int, 0] = []
    if request.resident:
        var cells = SpectralRBFCells(entry)
        var dsq = cells.allocate(max(nn, 1))
        var dx = cells.allocate(max(n * d, 1))
        var source: Array[Int, 1] = [request.source]
        var input_shape: Array[Int, 2] = [dx, n * d]
        _ = native_descriptor_call_fixed(entry, XD_DEV_UPLOAD, source, input_shape, no_floats)
        var sq: Array[Int, 7] = [dx, dx, dsq, n, n, d, 0]
        _ = native_descriptor_call_fixed(entry, XD_DEV_SQDIST, no_addresses, sq, distance)
        # One unused broadcast word, uploaded at the same stage as before.
        var zero = List[Float32](length=1, fill=0)
        var done = cells.allocate(1)
        var zero_address: Array[Int, 1] = [Int(zero.unsafe_ptr())]
        var zero_shape: Array[Int, 2] = [done, 1]
        _ = native_descriptor_call_fixed(entry, XD_DEV_UPLOAD, zero_address, zero_shape, no_floats)
        _ = zero
        var dscaled = cells.allocate(max(nn, 1))
        var scaling: Array[Int, 11] = [dsq, done, done, dscaled, 14, nn, n, 1, 3, 1, 3]
        _ = native_descriptor_call_fixed(entry, XD_DEV_EW, no_addresses, scaling, scale)
        var dexp = cells.allocate(max(nn, 1))
        var exponential: Array[Int, 11] = [dscaled, done, done, dexp, 9, nn, n, 1, 3, 1, 3]
        _ = native_descriptor_call_fixed(entry, XD_DEV_EW, no_addresses, exponential, scalar_zero)
        var destination: Array[Int, 1] = [request.destination]
        var output_shape: Array[Int, 2] = [dexp, nn]
        _ = native_descriptor_call_fixed(entry, XD_DEV_DOWNLOAD, destination, output_shape, no_floats)
        _ = cells^
        return
    var squared = List[Float32](length=nn, fill=0)
    var sq_addresses: Array[Int, 3] = [request.source, request.source, Int(squared.unsafe_ptr())]
    var sq_shape: Array[Int, 4] = [n, n, d, 0]
    _ = native_descriptor_call_fixed(entry, XD_SQDIST, sq_addresses, sq_shape, distance)
    var zero = List[Float32](length=1, fill=0)
    var scaled = List[Float32](length=nn, fill=0)
    var scale_addresses: Array[Int, 4] = [Int(squared.unsafe_ptr()), Int(zero.unsafe_ptr()),
        Int(zero.unsafe_ptr()), Int(scaled.unsafe_ptr())]
    var scale_shape: Array[Int, 7] = [14, nn, n, 1, 3, 1, 3]
    _ = native_descriptor_call_fixed(entry, XD_EW, scale_addresses, scale_shape, scale)
    _ = squared^
    var exp_addresses: Array[Int, 4] = [Int(scaled.unsafe_ptr()), Int(zero.unsafe_ptr()),
        Int(zero.unsafe_ptr()), request.destination]
    var exp_shape: Array[Int, 7] = [9, nn, n, 1, 3, 1, 3]
    _ = native_descriptor_call_fixed(entry, XD_EW, exp_addresses, exp_shape, scalar_zero)
    _ = scaled^
    _ = zero^
