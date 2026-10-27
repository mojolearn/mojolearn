# SPDX-License-Identifier: Apache-2.0
# Fixed private ABI metadata shared by provider admission and native engine.
comptime KERNEL_METHOD_ABI = "mojolearn.kernel_methods.native.v1"
comptime KERNEL_METHOD_OPERATIONS = 17


def kernel_method_operation(name: String) raises -> Int:
    if name == "kernel_ridge_fit":
        return 0
    if name == "kernel_ridge_predict":
        return 1
    if name == "kernel_ridge_fit_ptr":
        return 2
    if name == "kernel_ridge_predict_ptr":
        return 3
    if name == "kernel_ridge_resident_prepare":
        return 4
    if name == "kernel_ridge_resident_release":
        return 5
    if name == "kernel_ridge_predict_resident":
        return 6
    if name == "nystroem_fit_transform_ptr":
        return 7
    if name == "nystroem_fit":
        return 8
    if name == "nystroem_fit_ptr":
        return 9
    if name == "nystroem_transform":
        return 10
    if name == "rbf_sampler_fit":
        return 11
    if name == "rbf_sampler_transform":
        return 12
    raise Error("kernel methods provider: unknown operation name " + name)


def kernel_method_scalar_mask(operation: Int) raises -> Tuple[Int, Int]:
    if operation == 0:
        return (8, 224)
    if operation == 1:
        return (10, 224)
    if operation == 2:
        return (8, 224)
    if operation == 3:
        return (10, 96)
    if operation == 4:
        return (3, 0)
    if operation == 5:
        return (0, 0)
    if operation == 6:
        return (10, 96)
    if operation == 7:
        return (8, 48)
    if operation == 8:
        return (8, 48)
    if operation == 9:
        return (8, 48)
    if operation == 10:
        return (9, 48)
    if operation == 11:
        return (4, 4)
    if operation == 12:
        return (7, 52)
    raise Error("kernel methods provider: unknown operation schema")
