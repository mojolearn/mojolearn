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
// Metadata and the error buffer are CPU memory.
// Callers synchronize producers first. Execution and copies finish before return.
// Whole-model entries only: the per-operation entries (mojolearn_adapter_operation and
// mojolearn_adapter_operation_host_buffers) were removed on 2026-10-24 (docs/TOMBSTONES.md).
int64_t mojolearn_adapter_abi(void);
// Whole native Sequential training, additive ABI v1. Inputs: X, Y, parameters;
// resume adds moment1, moment2, state, previous curve. Outputs (five descriptors):
// parameters, moment1, moment2, state, curve. Inputs are never modified.
// State is 24 opaque int32 words (12 native Int64 counters/scalar bit patterns).
// Curve capacity = requested epochs + previous curve capacity (zero on fresh fit).
// Valid curve prefix length is Int64 state[1]; unused tail is zero.
// Model program: autograd/model.mojo version 1 or 2. Options (13 Int64s):
// loss,target,optimizer,flags,schedule,schedule_step,batch,epochs,shuffle,
// seed_low32,seed_high32,patience,initialize.
// Scalars are FLOAT64: lr,momentum/beta1,beta2,epsilon,
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
