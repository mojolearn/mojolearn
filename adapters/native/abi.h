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
// Whole native Sequential training, additive ABI v1. Inputs: X, Y, parameters;
// resume adds moment1, moment2, state, previous curve. Outputs (five descriptors):
// parameters, moment1, moment2, state, curve. Inputs are never modified.
// State is 24 opaque int32 words (12 native Int64 counters/scalar bit patterns).
// Curve capacity = requested epochs + previous curve capacity (zero on fresh fit).
// Valid curve prefix length is Int64 state[1]; unused tail is zero.
// Model program: autograd/model.mojo version 1 or 2. Options (13 Int64s):
// loss,target,optimizer,flags,schedule,schedule_step,batch,epochs,shuffle,
// seed_low32,seed_high32,patience,initialize.
// Scalars are FLOAT64 (unlike operation): lr,momentum/beta1,beta2,epsilon,
// weight_decay,dampening,tolerance,schedule_gamma,huber_delta, then one per layer.
// host_buffers=1 accepts host storage with GPU device=0; it refuses CPU -1.
// Application/framework autograd and optimizers are not part of this entry.
int64_t mojolearn_adapter_train(
    int64_t device, int64_t host_buffers, int64_t inputs, int64_t input_count,
    int64_t outputs, int64_t program, int64_t program_count,
    int64_t options, int64_t option_count, int64_t scalars, int64_t scalar_count,
    int64_t error, int64_t error_capacity);
#ifdef __cplusplus
}
#endif
