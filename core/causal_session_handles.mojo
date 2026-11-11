# SPDX-License-Identifier: Apache-2.0
"""The thin owner of a CausalLM model's native session handles (lane
causal-lm-sessions-native, 2026-11-11).

The Python object a model keeps for its resident sessions (`_resident_sessions`)
or its decode program (`_decode_program`) is this one native value: the
(tier, vendor) key, the session family of
`core/causal_lm_entry_table_sessions.mojo` and the handle words. It is created
when the sessions open (UNPACK of the generation call) and closes every handle
on `close()` or when Python drops it, as the Python-owned session objects
released their device state on collection. It holds no Python object and no
device state of its own.
"""
from core.causal_lm_entry_table import CausalEntryKey
from core.causal_lm_entry_table_sessions import causal_sessions_call, CAUSAL_SESSION_DECODE_CLOSE


struct CausalSessionHandles(Movable, Writable):
    var mode: Int
    var vendor: Int
    var family: Int
    var primary: Int
    """The model session (CAUSAL_SESSIONS_TRANSFORMER) or decode program
    handle word; 0 once closed."""
    var primary_close: Int
    """The operation that closes `primary` (LM_CLOSE or PROGRAM_RELEASE)."""
    var layers: List[Int]
    """One decode session handle word per Transformer layer, in order."""

    def __init__(out self, key: CausalEntryKey, family: Int, primary: Int, primary_close: Int):
        self.mode = key.mode
        self.vendor = key.vendor
        self.family = family
        self.primary = primary
        self.primary_close = primary_close
        self.layers = List[Int]()

    def key(self) -> CausalEntryKey:
        return CausalEntryKey(self.mode, self.vendor)

    def close(mut self) raises:
        """Close the model session first, then every layer session in order
        (the former Python close order). A refused close (a busy session)
        raises and keeps that handle and the ones after it."""
        if self.primary != 0:
            _ = causal_sessions_call(self.family, self.key(), self.primary_close, List[Int](), [self.primary], List[Float64]())
            self.primary = 0
        var closed = 0
        try:
            for i in range(len(self.layers)):  # driver-loop: over layer decode sessions
                _ = causal_sessions_call(self.family, self.key(), CAUSAL_SESSION_DECODE_CLOSE, List[Int](),
                    [self.layers[i]], List[Float64]())
                closed += 1
        finally:
            var rest = List[Int]()
            for i in range(closed, len(self.layers)):  # driver-loop: over unclosed layer handles
                rest.append(self.layers[i])
            self.layers = rest^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CausalSessionHandles(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __deinit__(deinit self):
        if self.primary == 0 and len(self.layers) == 0:
            return
        try:
            self.close()
        except:
            # Collection cannot report; a busy session is closed by its own
            # registry entry's destruction at process end.
            pass
