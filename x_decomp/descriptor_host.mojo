# SPDX-License-Identifier: Apache-2.0
# The x_decomp kernel-call table of the host binding (x_decomp/descriptor_schema.mojo),
# exported by bindings/_mojolearn_x_decomp_host.mojo as `x_decomp_native_api_v1`.
# The host image has no device pool: it serves the host-address operations on
# HostExec (the executor `ew_py[HostExec]` etc. use) and refuses the resident
# ones, as the host binding exports no `x_decomp_dev_*` entry.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from x_decomp.host import HostExec
from x_decomp.descriptor_ops import xd_address_op, xd_is_address_op
from x_decomp.descriptor_schema import XD_ABI_NAME, XD_MODE, XD_CAPABILITIES, XD_CAP_HOST, xd_validate_call


def x_decomp_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
                     ff: Int, nf: Int, result: Int, context: Int,
                     sink: NativeDescriptorError) abi("C") -> Int:
    try:
        xd_validate_call(operation, na, np, nf, result)
        var a = native_descriptor_words(aa, na)
        var p = native_descriptor_words(pp, np)
        var answer = 0
        if operation == XD_MODE:
            answer = GLOBAL_NUMERIC_MODE
        elif operation == XD_CAPABILITIES:
            answer = XD_CAP_HOST
        elif xd_is_address_op(operation):
            answer = xd_address_op[HostExec](operation, a, p, ff)
        else:
            raise Error("x_decomp native ABI: the host binding has no device-resident operation "
                        + String(operation))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct XDecompHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XD_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(x_decomp_host_v1)


# The extension image owns the table and name borrowed by PyCapsule. A named
# global is process-wide, so the name carries the image's numeric mode (as
# x_neighbors/device_ops.mojo names its context): the identical and fast
# images loaded together must each export their own dispatch function.
comptime _X_DECOMP_HOST_API_NAME = "MojoXDecompHostAPIIdenticalv1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXDecompHostAPIFastv1"
comptime X_DECOMP_HOST_API = _Global[StorageType=XDecompHostAPI,
    name=_X_DECOMP_HOST_API_NAME, init_fn=XDecompHostAPI.__init__]
