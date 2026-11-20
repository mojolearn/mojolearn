# SPDX-License-Identifier: Apache-2.0
"""Typed Mamba decode-session entry (lane py-lm-neural, 2026-11-14).

`Mamba1DecodeSession` / `Mamba2DecodeSession` / `Mamba3DecodeSession`'s host
arm on typed values: the session owns its weight and state copies in a
`MambaDecodePlan` (mamba/impl/mamba_decode_plan.mojo) and runs each token
through the block's carried-state image (MAMBA_STATE_ABI_NAME, op
2 * kind - 1: the per-call state step), exactly as the public door's host arm
does (`bindings/mamba_public_native_door.mojo` `_session_step_call`,
`mamba_session_decode`), with no Python object, no provider capsule and no
door. The caller resolves the image once (`Mamba3StateRoute.state_ref()`,
core/mamba_state_entry_table.mojo) and names every buffer by address; native
checks call this instead of rebuilding door glue.

Parallel plan: one token is one image call (the image's own kernel); the
loops here are the decode driver over tokens and the per-row copy of token t
into and out of the (B, d) buffers, the door's host-arm loops, moved.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call
from core.hostptr import f32_ptr
from mamba.impl.mamba_decode_plan import MambaDecodePlan


struct MambaDecodeSessionRun(Movable):
    """An open host-arm decode session.

    `kind` 1, 2 or 3. `weights`: the block's weights in its ABI order, copied
    in. `parts`: the state's named pieces (the plan's `names` order), copied
    in; `dims` their shapes; `report_dims` the reports' shapes (Mamba-1 none,
    Mamba-2 one, Mamba-3 four). `q`, `pending`: Mamba-2/3's buffered count
    and Mamba-3's pending flag; `position`, `profile`: Mamba-1's absolute
    position and numerical profile. `dt_low`, `dt_high`: Mamba-2's dt limit."""

    var plan: MambaDecodePlan
    var entry: EntryRef
    var dt_low: Float64
    var dt_high: Float64

    def __init__(out self, kind: Int, entry: EntryRef, b: Int, d_model: Int,
                 var weights: List[List[Float32]], var parts: List[List[Float32]],
                 var names: List[String], var dims: List[List[Int]], var report_dims: List[List[Int]],
                 q: Int = 0, pending: Bool = False, position: Int = 0, profile: Int = 1,
                 dt_low: Float64 = 0.0, dt_high: Float64 = 0.0) raises:
        if kind < 1 or kind > 3:
            raise Error("TypeError: a Mamba decode session takes a Mamba1Block, Mamba2Block or Mamba3Block")
        if b <= 0 or d_model <= 0:
            raise Error("Mamba decode session: batch and d_model must be positive")
        if len(parts) != len(dims) or len(names) != len(dims):
            raise Error("Mamba decode session: one shape and one name per state piece")
        for i in range(len(parts)):  # driver-loop: over the state's named pieces (admission)
            var count = 1
            for axis in range(len(dims[i])):  # small-loop(state piece shape rank)
                count *= dims[i][axis]
            if len(parts[i]) != count:
                raise Error("Mamba decode session: state piece " + names[i] + " has the wrong size")
        self.plan = MambaDecodePlan(kind)
        self.plan.b = b
        self.plan.d_model = d_model
        self.plan.q = q
        self.plan.pending = pending
        self.plan.position = position
        self.plan.profile = profile
        self.plan.names = names^
        self.plan.dims = dims^
        self.plan.report_dims = report_dims^
        self.plan.weights = weights^
        self.plan.parts = parts^
        self.plan.open = True
        self.entry = entry
        self.dt_low = dt_low
        self.dt_high = dt_high

    def _require_open(self, what: String) raises:
        if not self.plan.open:
            raise Error(what + ": the session is closed")

    def report_count(self, index: Int) -> Int:
        var n = 1
        for axis in range(len(self.plan.report_dims[index])):  # small-loop(report shape rank)
            n *= self.plan.report_dims[index][axis]
        return n

    def step(mut self, x: Int, y: Int, reports: List[Int]) raises:
        """One token per row: x and y float32 (B, d_model); `reports` the
        report buffers' addresses (`report_dims`), written whole."""
        var what = "Mamba" + String(self.plan.kind) + "DecodeSession.step"
        self._require_open(what)
        if x == 0 or y == 0 or len(reports) != len(self.plan.report_dims):
            raise Error(what + ": null input or output, or a missing report buffer")
        var kind = self.plan.kind
        var addresses = self.plan.step_addresses(False, x, y, reports)
        var words = self.plan.step_parameters()
        var floats = self.plan.step_floats(self.dt_low, self.dt_high)
        var updated = entry_ref_call(self.entry, 2 * kind - 1, addresses, words, floats)
        if kind != 1:
            self.plan.q = updated
        if kind == 3:
            self.plan.pending = False
        if kind == 1 and self.plan.profile == 2:
            self.plan.position += 1

    def decode(mut self, x: Int, tokens: Int, y: Int, reports: List[Int]) raises:
        """`tokens` tokens per row in one call: x and y float32
        (B, tokens, d_model); y and the state equal `tokens` `step` calls
        byte for byte; the reports are the last token's."""
        var what = "Mamba" + String(self.plan.kind) + "DecodeSession.decode"
        self._require_open(what)
        if x == 0 or y == 0 or tokens < 0:
            raise Error(what + ": null input or output, or a negative token count")
        var b = self.plan.b
        var dm = self.plan.d_model
        var xt = List[Float32](length=b * dm, fill=Float32(0.0))
        var yt = List[Float32](length=b * dm, fill=Float32(0.0))
        var src = f32_ptr(x)
        var dst = f32_ptr(y)
        for t in range(tokens):  # driver-loop: over decode tokens, one host-arm step each
            for r in range(b):  # driver-loop: over the session's batch rows, a host-arm copy of token t
                unsafe_memcpy(dest=xt.unsafe_ptr() + r * dm, src=src + (r * tokens + t) * dm, count=dm)
            self.step(Int(xt.unsafe_ptr()), Int(yt.unsafe_ptr()), reports)
            for r in range(b):  # driver-loop: over the session's batch rows, a host-arm copy of token t
                unsafe_memcpy(dest=dst + (r * tokens + t) * dm, src=yt.unsafe_ptr() + r * dm, count=dm)
        _ = xt^
        _ = yt^

    def state_address(self, piece: Int) -> Int:
        """The address of the session's own copy of state piece `piece`."""
        return Int(self.plan.parts[piece].unsafe_ptr())

    def close(mut self):
        self.plan.open = False
