// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// ABI v1 uses 64-bit host words. Each tensor descriptor has eleven words:
// storage address, kind (0=float32, 1=int32), rank, eight dimensions.
// All input/output storage belongs to the caller and remains valid until return.
// Tensor storage is contiguous, positive-sized, on CPU (-1) or visible GPU 0.
// Metadata and the error buffer are CPU memory. Operation codes and options are
// those of autograd/ops_api.mojo, with tensor-only embedding/take index forms.
// Value inputs are float32. Int32 inputs are admitted only as embedding/take
// indices or for OP_COPY; argmax may produce int32. This is not integer math.
// The output descriptor is checked against the actual result before copying.
// Callers synchronize producers first. Execution and copies finish before return.
int64_t mojolearn_adapter_abi(void);
int64_t mojolearn_adapter_operation(
    int64_t device, int64_t code, int64_t inputs, int64_t input_count,
    int64_t output, int64_t integer_args, int64_t integer_count,
    int64_t scalar_args, int64_t scalar_count, int64_t error, int64_t error_capacity);
// Additive ABI v1 entry: tensor descriptors name HOST buffers, but device must
// be 0 and every numerical operation runs through DeviceExec on that GPU.
// Inputs are uploaded and the result downloaded with native byte copies.
// Callers keep host buffers alive until return; copies and GPU work are drained.
// CPU device -1 is refused, so this entry cannot silently run math on the CPU.
int64_t mojolearn_adapter_operation_host_buffers(
    int64_t device, int64_t code, int64_t inputs, int64_t input_count,
    int64_t output, int64_t integer_args, int64_t integer_count,
    int64_t scalar_args, int64_t scalar_count, int64_t error, int64_t error_capacity);
#ifdef __cplusplus
}
#endif
