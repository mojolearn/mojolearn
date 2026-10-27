// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

// Private native execution ABI, distinct from the whole-model framework API
// in adapters/native/abi.h. This does NOT restore per-operation Torch/MAX ops.
// All supported hosts use 64-bit words. The versioned capsule contains one
// pointer-sized slot of type ml_generation_session_forward_v1. Consumers copy
// the function pointer once and retain the owning module and capsule.
// Capsule name: mojolearn.transformer.session_forward.v1
//
// The session handle is obtained by the owning extension's checked admission
// entry and is opaque. Retain that session object for every use of the handle;
// do not dereference it, transfer it to another image, or use it after release.
// Inputs are host-resident native int64 vectors, with the existing transformer's
// 13 (+11) buffer addresses and 10 (+17) scalar/options words. These vectors
// and ALL referenced buffers remain valid through return. No pointers are
// retained by this call. Weights/cache mutation behavior is unchanged.
//
// Phase 0 reserves the session (checks open/busy, sets busy) while the provider
// holds its interpreter GIL. Phase 1 executes the supplied descriptors with
// that GIL released. Phase 2 releases the reservation after reacquiring GIL,
// on success AND failure (including failure to enter the GIL release scope). Native
// callers without an interpreter must provide equivalent serialization.
// Do not abandon a successful phase 0: invoke phase 1 then phase 2, or phase
// 2 alone to cancel before execution. No other caller may execute phase 2.
// Execution uses only native values. This entry neither calls CPython nor invokes Python callbacks.
// The operation and its required synchronization finish before return.
// 0 = success, writes cached_tokens to result. 1 = failure, calls error_sink
// once with complete UTF-8 bytes (not NUL-terminated), borrowed during that
// call. error_sink is a compiled native function, must not raise, and must copy
// bytes if retaining them. This avoids truncation, rerunning failed work, and
// cross-image allocation/free. Validation/cleanup match the Python entry.
typedef void (*ml_generation_error_sink_v1)(
    int64_t context, int64_t utf8_bytes, int64_t byte_count);
typedef int64_t (*ml_generation_session_forward_v1)(
    int64_t phase, int64_t session, int64_t addresses, int64_t address_count,
    int64_t parameters, int64_t parameter_count, int64_t result,
    int64_t error_context, ml_generation_error_sink_v1 error_sink);
