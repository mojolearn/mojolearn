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
// The output descriptor is checked against the actual result before copying.
// Callers synchronize producers first. Execution and copies finish before return.
int64_t mojolearn_adapter_abi(void);
int64_t mojolearn_adapter_operation(
    int64_t device, int64_t code, int64_t inputs, int64_t input_count,
    int64_t output, int64_t integer_args, int64_t integer_count,
    int64_t scalar_args, int64_t scalar_count, int64_t error, int64_t error_capacity);
#ifdef __cplusplus
}
#endif
