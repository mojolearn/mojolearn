# SPDX-License-Identifier: Apache-2.0
# Fixed private ABI metadata shared by provider admission and native engine.
comptime KERNEL_METHOD_ABI = "mojolearn.kernel_methods.native.v1"
comptime KERNEL_METHOD_OPERATIONS = 17


# Operation ids (the by-name lookup `kernel_method_operation` is gone: the door
# names an operation by its id, kernel_methods/native_call_run.mojo `KM_OP_*`):
# 0 kernel_ridge_fit, 1 kernel_ridge_predict, 2 kernel_ridge_fit_ptr,
# 3 kernel_ridge_predict_ptr, 4 kernel_ridge_resident_prepare,
# 5 kernel_ridge_resident_release, 6 kernel_ridge_predict_resident,
# 7 nystroem_fit_transform_ptr, 8 nystroem_fit, 9 nystroem_fit_ptr,
# 10 nystroem_transform, 11 rbf_sampler_fit, 12 rbf_sampler_transform;
# 13 to 16 are the metadata words and the scale-gamma limbs.


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
