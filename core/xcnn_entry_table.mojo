# SPDX-License-Identifier: Apache-2.0
"""Tier/vendor-qualified native table of the convolution family (lane
calltable-xcnn, 2026-11-11).

Modeled on `core/trees_entry_table.mojo`: the same `core/entry_table.mojo`
publication and lookup, separate slot names. Both x_cnn images
(`_mojolearn_x_cnn`, `_mojolearn_x_cnn_host`) publish one entry at module
init, `x_cnn_entry_v1`, a dispatch over the operations below
(`x_cnn/entry_runs.mojo`, `x_cnn/host/entry_runs_host.mojo`). The CNN
layer doors (`bindings/xcnn_*.mojo`) read the slot of the image the layer's
tier connected (the bridge records its key as `native_key`) and call the
pointer with plain spans; no Python object crosses.

A call is the descriptor shape of `core/native_descriptor_abi.mojo`:
`operation` (an `XC_*` code), a span of addresses, a span of integers and a
span of Float64 scalars (only the dropout operations read one: the drop
probability). The addresses and integers are those of the operation's former
Python entry (`x_cnn_<name>`), in its order: the addresses (host addresses
or `x_cnn_res_alloc` handles) as listed, then the integer parameters. A `_m`
operation's integers start with its residency word (`dev`), then its
parameters. The resident-array operations take the handle first:
RES_ALLOC / RES_ALLOC_NOFILL ints [n] -> handle; RES_FREE addrs [h];
RES_UPLOAD addrs [h, src] ints [n]; RES_DOWNLOAD addrs [h, dst] ints [n];
CSR_UPLOAD addrs [csr] ints [n, F, nnz, mode] -> handle. The result is the
count the Python entry returned (a handle for the allocations, K for
GCN_LOOPS). The caller owns every buffer until the synchronous call returns.

An image without an operation (the host image has no `_m` operation and no
CSR_UPLOAD) refuses it with the text a missing Python entry raised. Adding an
operation appends a code and bumps nothing; changing a layout needs a new V
suffix. Publication and lookup are small metadata operations with no
independent numerical work to split.
"""
from core.entry_table import entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call

comptime XC_ENTRY_ABI = "mojolearn.cnn.x_cnn.entry.v1"

comptime XC_RES_ALLOC = 0
comptime XC_RES_ALLOC_NOFILL = 1
comptime XC_RES_FREE = 2
comptime XC_RES_UPLOAD = 3
comptime XC_RES_DOWNLOAD = 4
comptime XC_GEMM = 5
comptime XC_CONV2D_FORWARD = 6
comptime XC_CONV2D_BACKWARD = 7
comptime XC_MAXPOOL2D_FORWARD = 8
comptime XC_MAXPOOL2D_BACKWARD = 9
comptime XC_AVGPOOL2D_FORWARD = 10
comptime XC_AVGPOOL2D_BACKWARD = 11
comptime XC_RELU_FORWARD = 12
comptime XC_RELU_BACKWARD = 13
comptime XC_ADD = 14
comptime XC_MUL = 15
comptime XC_LINEAR_FORWARD = 16
comptime XC_LINEAR_BACKWARD = 17
comptime XC_BATCHNORM_FORWARD = 18
comptime XC_BATCHNORM_BACKWARD = 19
comptime XC_DROPOUT2D = 20
comptime XC_SPMM = 21
comptime XC_GCN_NORM = 22
comptime XC_PAD2D_FORWARD = 23
comptime XC_PAD2D_BACKWARD = 24
comptime XC_ADAPTIVE_POOL = 25
comptime XC_GRAPH_OP = 26
comptime XC_CSR_BUILD = 27
comptime XC_CSR_UPLOAD = 28
comptime XC_PERM_GATHER = 29
comptime XC_VALS_T = 30
comptime XC_GCN_LOOPS = 31
comptime XC_GCN_BACKWARD_T = 32
comptime XC_SAGE_BACKWARD_T = 33
comptime XC_GEMM_M = 34
comptime XC_CONV2D_FORWARD_M = 35
comptime XC_CONV2D_BACKWARD_M = 36
comptime XC_MAXPOOL2D_FORWARD_M = 37
comptime XC_MAXPOOL2D_BACKWARD_M = 38
comptime XC_AVGPOOL2D_FORWARD_M = 39
comptime XC_AVGPOOL2D_BACKWARD_M = 40
comptime XC_MAP2_M = 41
comptime XC_LINEAR_FORWARD_M = 42
comptime XC_BATCHNORM_FORWARD_M = 43
comptime XC_BATCHNORM_BACKWARD_M = 44
comptime XC_DROPOUT2D_M = 45
comptime XC_SPMM_M = 46
comptime XC_ADAPTIVE_POOL_M = 47
comptime XC_GRAPH_OP_M = 48
comptime XC_GCN_NORM_M = 49
comptime XC_PAD2D_M = 50
comptime XC_CHAN_COPY_M = 51
comptime XC_OP_COUNT = 52


@fieldwise_init
struct XcnnEntryKey(Copyable, Movable):
    var mode: Int
    var vendor: Int


def xcnn_entry_vendor(vendor: StaticString) -> Int:
    """The vendor word of the key for the vendor string an x_cnn image
    reports as `x_cnn_vendor` ("cpu" for the host image, `COMPILED_VENDOR`
    for the GPU one): 0 cpu, 1 metal, 2 cuda, 3 hip, else -1. The bridge
    (`python/mojolearn/_native_xcnn_bridge.py`) applies the same mapping."""
    if vendor == "cpu":
        return 0
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def xcnn_op_name(op: Int) -> String:
    """The former Python entry name of an operation (refusal text)."""
    if op == XC_RES_ALLOC:
        return "x_cnn_res_alloc"
    if op == XC_RES_ALLOC_NOFILL:
        return "x_cnn_res_alloc_nofill"
    if op == XC_RES_FREE:
        return "x_cnn_res_free"
    if op == XC_RES_UPLOAD:
        return "x_cnn_res_upload"
    if op == XC_RES_DOWNLOAD:
        return "x_cnn_res_download"
    if op == XC_GEMM:
        return "x_cnn_gemm"
    if op == XC_CONV2D_FORWARD:
        return "x_cnn_conv2d_forward"
    if op == XC_CONV2D_BACKWARD:
        return "x_cnn_conv2d_backward"
    if op == XC_MAXPOOL2D_FORWARD:
        return "x_cnn_maxpool2d_forward"
    if op == XC_MAXPOOL2D_BACKWARD:
        return "x_cnn_maxpool2d_backward"
    if op == XC_AVGPOOL2D_FORWARD:
        return "x_cnn_avgpool2d_forward"
    if op == XC_AVGPOOL2D_BACKWARD:
        return "x_cnn_avgpool2d_backward"
    if op == XC_RELU_FORWARD:
        return "x_cnn_relu_forward"
    if op == XC_RELU_BACKWARD:
        return "x_cnn_relu_backward"
    if op == XC_ADD:
        return "x_cnn_add"
    if op == XC_MUL:
        return "x_cnn_mul"
    if op == XC_LINEAR_FORWARD:
        return "x_cnn_linear_forward"
    if op == XC_LINEAR_BACKWARD:
        return "x_cnn_linear_backward"
    if op == XC_BATCHNORM_FORWARD:
        return "x_cnn_batchnorm_forward"
    if op == XC_BATCHNORM_BACKWARD:
        return "x_cnn_batchnorm_backward"
    if op == XC_DROPOUT2D:
        return "x_cnn_dropout2d"
    if op == XC_SPMM:
        return "x_cnn_spmm"
    if op == XC_GCN_NORM:
        return "x_cnn_gcn_norm"
    if op == XC_PAD2D_FORWARD:
        return "x_cnn_pad2d_forward"
    if op == XC_PAD2D_BACKWARD:
        return "x_cnn_pad2d_backward"
    if op == XC_ADAPTIVE_POOL:
        return "x_cnn_adaptive_pool"
    if op == XC_GRAPH_OP:
        return "x_cnn_graph_op"
    if op == XC_CSR_BUILD:
        return "x_cnn_csr_build"
    if op == XC_CSR_UPLOAD:
        return "x_cnn_csr_upload"
    if op == XC_PERM_GATHER:
        return "x_cnn_perm_gather"
    if op == XC_VALS_T:
        return "x_cnn_vals_t"
    if op == XC_GCN_LOOPS:
        return "x_cnn_gcn_loops"
    if op == XC_GCN_BACKWARD_T:
        return "x_cnn_gcn_backward_t"
    if op == XC_SAGE_BACKWARD_T:
        return "x_cnn_sage_backward_t"
    if op == XC_GEMM_M:
        return "x_cnn_gemm_m"
    if op == XC_CONV2D_FORWARD_M:
        return "x_cnn_conv2d_forward_m"
    if op == XC_CONV2D_BACKWARD_M:
        return "x_cnn_conv2d_backward_m"
    if op == XC_MAXPOOL2D_FORWARD_M:
        return "x_cnn_maxpool2d_forward_m"
    if op == XC_MAXPOOL2D_BACKWARD_M:
        return "x_cnn_maxpool2d_backward_m"
    if op == XC_AVGPOOL2D_FORWARD_M:
        return "x_cnn_avgpool2d_forward_m"
    if op == XC_AVGPOOL2D_BACKWARD_M:
        return "x_cnn_avgpool2d_backward_m"
    if op == XC_MAP2_M:
        return "x_cnn_map2_m"
    if op == XC_LINEAR_FORWARD_M:
        return "x_cnn_linear_forward_m"
    if op == XC_BATCHNORM_FORWARD_M:
        return "x_cnn_batchnorm_forward_m"
    if op == XC_BATCHNORM_BACKWARD_M:
        return "x_cnn_batchnorm_backward_m"
    if op == XC_DROPOUT2D_M:
        return "x_cnn_dropout2d_m"
    if op == XC_SPMM_M:
        return "x_cnn_spmm_m"
    if op == XC_ADAPTIVE_POOL_M:
        return "x_cnn_adaptive_pool_m"
    if op == XC_GRAPH_OP_M:
        return "x_cnn_graph_op_m"
    if op == XC_GCN_NORM_M:
        return "x_cnn_gcn_norm_m"
    if op == XC_PAD2D_M:
        return "x_cnn_pad2d_m"
    if op == XC_CHAN_COPY_M:
        return "x_cnn_chan_copy_m"
    return "x_cnn operation " + String(op)


def _xcnn_entry_key_check(key: XcnnEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 0 or key.vendor > 3:
        raise Error("native x_cnn entries: invalid tier/vendor key")


def xcnn_table_publish(key: XcnnEntryKey, publisher: Int, schemas: List[String], operations: List[Int],
                       entries: List[NativeDescriptorCall]) raises:
    _xcnn_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastCPUV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastMetalV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastCUDAV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastHIPV1"
        entry_table_publish[NAME](key.mode, key.vendor, publisher, schemas, operations, entries)
        return
    raise Error("native x_cnn entries: unsupported execution key")


def xcnn_table_entry(key: XcnnEntryKey, index: Int, schema: String, operations: Int) raises -> NativeDescriptorCall:
    _xcnn_entry_key_check(key)
    if key.vendor == 0 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 0 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastCPUV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 1 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastMetalV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 2 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastCUDAV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 1:
        comptime NAME = "MojolearnEntriesXCnnIdenticalHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    if key.vendor == 3 and key.mode == 0:
        comptime NAME = "MojolearnEntriesXCnnFastHIPV1"
        return entry_table_read[NAME](key.mode, key.vendor, index, schema, operations)
    raise Error("native x_cnn entries: unsupported execution key")


def xcnn_entry_call(key: XcnnEntryKey, op: Int, addrs: List[Int], ints: List[Int],
                    floats: List[Float64]) raises -> Int:
    """One x_cnn operation in the image the key names: the published entry
    read once (an acquire load and a schema compare), then the call. The
    caller keeps every addressed buffer alive and releases the GIL when it
    holds one."""
    var entry = xcnn_table_entry(key, 0, String(XC_ENTRY_ABI), XC_OP_COUNT)
    return native_descriptor_call(entry, op, addrs, ints, floats)
