# SPDX-License-Identifier: Apache-2.0
"""The CausalLM session tables: GPU session state behind native registry
handles (lane causal-lm-sessions-native, 2026-11-11).

The resident generation sessions (one `TransformerDecodeSession` per layer and
the model's `CausalLMSession`) and the decode program (`CausalLMProgram`) are
owned by their GPU image in `core/state_registry.mojo` registries
(generation-checked handles, Arc leases). A caller holds only handle words and
reaches every session operation through one descriptor entry the owning image
publishes at module init, keyed by the model's (tier, vendor) like
`core/causal_lm_entry_table.mojo`:

  CAUSAL_SESSIONS_TRANSFORMER  `transformer/causal_session_entry.mojo::
                               causal_sessions_transformer_v1`
                               (`_mojolearn_transformer`): every operation.
  CAUSAL_SESSIONS_MAMBA        `mamba/causal_session_entry.mojo::
                               causal_sessions_mamba_v1` (`_mojolearn_mamba`):
                               the decode program operations only.

Both entries share one schema and operation numbering, so one reader serves
both. A handle word is `generation << 32 | index` (never 0: generations start
at 1); 0 means "no handle". GPU keys only (vendor 1 metal, 2 cuda, 3 hip): the
CPU images own no resident session.

Publication and lookup are small metadata operations with no independent
numerical work to split.
"""
from core.entry_table import EntryRef, entry_ref_call, entry_ref_word, entry_table_publish, entry_table_read
from core.native_descriptor_abi import NativeDescriptorCall
from core.causal_lm_entry_table import CausalEntryKey
from core.state_registry import StateHandle

comptime CAUSAL_SESSIONS_TRANSFORMER = 0
comptime CAUSAL_SESSIONS_MAMBA = 1

comptime CAUSAL_SESSIONS_SCHEMA = "mojolearn.causal.sessions.v1"
comptime CAUSAL_SESSIONS_OPERATIONS = 17

# Operation codes. Words are [handle, scalars...] unless stated.
comptime CAUSAL_SESSION_DECODE_CREATE = 0     # no words -> handle
comptime CAUSAL_SESSION_DECODE_OPEN = 1       # a: 11 (+ options tail); words [h, 9 scalars (+ tail)] -> cached
comptime CAUSAL_SESSION_DECODE_OPEN15 = 2     # a: 11 + tail + 21 planes; words [h, 9 + tail] -> cached
comptime CAUSAL_SESSION_DECODE_EXPORT = 3     # a: [k, v]; words [h] -> cached
comptime CAUSAL_SESSION_DECODE_LOAD = 4       # a: [k, v]; words [h, cached] -> cached
comptime CAUSAL_SESSION_DECODE_CLOSE = 5      # words [h]: close and release the handle
comptime CAUSAL_SESSION_LM_CREATE = 6         # no words -> handle
comptime CAUSAL_SESSION_LM_OPEN = 7           # a: [embed, norm, head]; words [h, B, V, D]; floats [eps]
comptime CAUSAL_SESSION_LM_OPEN15 = 8         # a: [embed, norm, hi, lo, exponents]; words [h, B, V, D]; floats [eps]
comptime CAUSAL_SESSION_LM_RUN = 9            # a: [ids, ids_out, logits or 0]; words [h, n, layer handles..., L, steps, cached]
comptime CAUSAL_SESSION_LM_FORWARD = 10       # a: [ids, logits]; words [h, n, layer handles..., L]
comptime CAUSAL_SESSION_LM_CLOSE = 11         # words [h]: close and release the handle
comptime CAUSAL_SESSION_PROGRAM_READ = 12     # a: read descriptor; words [h or 0, out, capacity, p...]; floats f -> handle
comptime CAUSAL_SESSION_PROGRAM_RELEASE = 13  # words [h]
# Lane fix-transformer-decode-door (2026-11-13): the public decode session's
# per-token step, its resident N-token decode and its L-token forward on a
# registry decode session, so the public door reaches them natively (no Python
# call into the GPU binding per token). The position is the caller's belief,
# refused on a mismatch as the former session entries refused it.
# Hot-path audit 2026-11-13 (BLOCK causal_session_entry.mojo:165): the door
# resolves its session ONCE, at open (PIN: a lease on the session, held by the
# door's handle until the handle is destroyed), and every step / decode /
# forward runs on the pinned session (PINNED): no registry lookup or lock, no
# native List built per token. UNPIN drops the lease.
comptime CAUSAL_SESSION_DECODE_PIN = 14       # words [h] -> pin word (a lease on the session)
comptime CAUSAL_SESSION_DECODE_UNPIN = 15     # words [pin]: drop the lease
comptime CAUSAL_SESSION_DECODE_PINNED = 16    # a: [x, y]; words [pin, kind, count, cached] -> cached
#: `kind` of CAUSAL_SESSION_DECODE_PINNED (count is unused by a step).
comptime CAUSAL_DECODE_PINNED_STEP = 0        # x, y: B * d_model
comptime CAUSAL_DECODE_PINNED_RUN = 1         # x, y: B * N * d_model, count N
comptime CAUSAL_DECODE_PINNED_FORWARD = 2     # x, y: B * L * d_model, count L


def causal_session_handle_word(handle: StateHandle) raises -> Int:
    """One plain word for a registry handle (index low, generation high)."""
    if handle.index < 0 or handle.index > 0x7FFFFFFF or handle.generation == 0 or handle.generation > UInt64(0x7FFFFFFF):
        raise Error("native CausalLM sessions: handle out of range")
    return (Int(handle.generation) << 32) | handle.index


def causal_session_handle_of(word: Int) raises -> StateHandle:
    if word <= 0:
        raise Error("native CausalLM sessions: no session handle")
    return StateHandle(word & 0xFFFFFFFF, UInt64(word >> 32))


def _sessions_key_check(key: CausalEntryKey) raises:
    if key.mode < 0 or key.mode > 1 or key.vendor < 1 or key.vendor > 3:
        raise Error("native CausalLM sessions: invalid tier/vendor key (GPU images only)")


def _sessions_publish_named[name: StaticString](key: CausalEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    entry_table_publish[name](key.mode, key.vendor, publisher, [String(CAUSAL_SESSIONS_SCHEMA)],
        [CAUSAL_SESSIONS_OPERATIONS], [entry])


def _sessions_read_named[name: StaticString](key: CausalEntryKey) raises -> NativeDescriptorCall:
    return entry_table_read[name](key.mode, key.vendor, 0, String(CAUSAL_SESSIONS_SCHEMA), CAUSAL_SESSIONS_OPERATIONS)


def causal_sessions_publish[family: Int](key: CausalEntryKey, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish this image's session entry of `family` under `key` (module
    init). A repeated init publishes the same pointer."""
    comptime assert family == CAUSAL_SESSIONS_TRANSFORMER or family == CAUSAL_SESSIONS_MAMBA, "unknown CausalLM session family"
    _sessions_key_check(key)
    comptime if family == CAUSAL_SESSIONS_TRANSFORMER:
        if key.mode == 1 and key.vendor == 1:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerIdenticalMetalV1"](key, publisher, entry)
        elif key.mode == 0 and key.vendor == 1:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerFastMetalV1"](key, publisher, entry)
        elif key.mode == 1 and key.vendor == 2:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerIdenticalCUDAV1"](key, publisher, entry)
        elif key.mode == 0 and key.vendor == 2:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerFastCUDAV1"](key, publisher, entry)
        elif key.mode == 1 and key.vendor == 3:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerIdenticalHIPV1"](key, publisher, entry)
        else:
            _sessions_publish_named["MojolearnEntriesCausalSessionsTransformerFastHIPV1"](key, publisher, entry)
    else:
        if key.mode == 1 and key.vendor == 1:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaIdenticalMetalV1"](key, publisher, entry)
        elif key.mode == 0 and key.vendor == 1:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaFastMetalV1"](key, publisher, entry)
        elif key.mode == 1 and key.vendor == 2:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaIdenticalCUDAV1"](key, publisher, entry)
        elif key.mode == 0 and key.vendor == 2:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaFastCUDAV1"](key, publisher, entry)
        elif key.mode == 1 and key.vendor == 3:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaIdenticalHIPV1"](key, publisher, entry)
        else:
            _sessions_publish_named["MojolearnEntriesCausalSessionsMambaFastHIPV1"](key, publisher, entry)


def causal_sessions_entry(family: Int, key: CausalEntryKey) raises -> NativeDescriptorCall:
    """The published session entry of `family` for `key`; a missing or
    mismatched publication refuses with the table's error."""
    _sessions_key_check(key)
    if family == CAUSAL_SESSIONS_TRANSFORMER:
        if key.mode == 1 and key.vendor == 1:
            return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerIdenticalMetalV1"](key)
        if key.mode == 0 and key.vendor == 1:
            return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerFastMetalV1"](key)
        if key.mode == 1 and key.vendor == 2:
            return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerIdenticalCUDAV1"](key)
        if key.mode == 0 and key.vendor == 2:
            return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerFastCUDAV1"](key)
        if key.mode == 1 and key.vendor == 3:
            return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerIdenticalHIPV1"](key)
        return _sessions_read_named["MojolearnEntriesCausalSessionsTransformerFastHIPV1"](key)
    if family == CAUSAL_SESSIONS_MAMBA:
        if key.mode == 1 and key.vendor == 1:
            return _sessions_read_named["MojolearnEntriesCausalSessionsMambaIdenticalMetalV1"](key)
        if key.mode == 0 and key.vendor == 1:
            return _sessions_read_named["MojolearnEntriesCausalSessionsMambaFastMetalV1"](key)
        if key.mode == 1 and key.vendor == 2:
            return _sessions_read_named["MojolearnEntriesCausalSessionsMambaIdenticalCUDAV1"](key)
        if key.mode == 0 and key.vendor == 2:
            return _sessions_read_named["MojolearnEntriesCausalSessionsMambaFastCUDAV1"](key)
        if key.mode == 1 and key.vendor == 3:
            return _sessions_read_named["MojolearnEntriesCausalSessionsMambaIdenticalHIPV1"](key)
        return _sessions_read_named["MojolearnEntriesCausalSessionsMambaFastHIPV1"](key)
    raise Error("native CausalLM sessions: unknown session family")


def causal_sessions_ref(family: Int, key: CausalEntryKey) raises -> EntryRef:
    """The session entry of `family` in the image `key` names, resolved once
    (lane entry-table-zero-alloc); the table refusals raise here."""
    return EntryRef(causal_sessions_entry(family, key), entry_ref_word(key.mode, key.vendor, CAUSAL_SESSIONS_OPERATIONS))


def causal_sessions_call(entry: EntryRef, operation: Int, addresses: List[Int],
    words: List[Int], floats: List[Float64]) raises -> Int:
    """One synchronous session call through a resolved entry: one indirect
    call. The caller owns every buffer the addresses name until this returns."""
    return entry_ref_call(entry, operation, addresses, words, floats)


def causal_sessions_tables_admit(key: CausalEntryKey, kind_family: Int) raises:
    """Construction-time admission: the session entry the model's kind will
    call is published for its GPU key."""
    _ = causal_sessions_entry(kind_family, key)
