# SPDX-License-Identifier: Apache-2.0
# Internal, additive native ABI v1. No Python objects or Mojo struct layout
# cross this boundary. The capsule is only the setup-time table transport.
# See session_abi.h for ownership, synchronization and error rules.
comptime GENERATION_SESSION_ABI_NAME = "mojolearn.transformer.session_forward.v1"
comptime GenerationErrorSink = def(Int, Int, Int) thin abi("C") -> NoneType
comptime GenerationSessionForward = def(
    Int, Int, Int, Int, Int, Int, Int, Int, GenerationErrorSink,
) thin abi("C") -> Int
