# SPDX-License-Identifier: Apache-2.0
# Historical host-only mock orchestration for test_byte_lm_surface. Product
# execution now uses ByteHostSession and the shared compiled descriptor table.
import ctypes
import operator
from mojolearn._bufcheck import memcopy
from mojolearn._byte_lm_config import ByteLanguageModelConfig
_MAX_COMPLETED = 999999

def _index(value, what):
    if isinstance(value, bool) or type(value).__name__ in ("bool", "bool_"):
        raise ValueError(f"byte LM: {what} must be an integer, not a boolean")
    return operator.index(value)


def _config(shape):
    """`_byte_config`: 7 or 9 integers (B, L, DM, H, KV, HD, FF[, layers,
    vocab]), validated."""
    if len(shape) not in (7, 9):
        raise ValueError("byte LM: expected 7 or 9 shape integers (B,L,DM,H,KV,HD,FF[,layers,vocab])")
    values = [_index(v, "shape dimensions") for v in shape]
    if len(values) == 7:
        values += [2, 256]
    return ByteLanguageModelConfig(*values)


def _scatter_named(flat_address, named, cfg):
    """`byte_scatter_named`: the flat gradient copied into one output per
    parameter tensor, in registry order. A copy, no arithmetic."""
    if len(named) != cfg.n_tensors:
        raise ValueError("byte LM: expected one named output per parameter tensor")
    offsets = cfg.offsets
    for j, target in enumerate(named):  # driver-loop: over model parameter tensors
        count = offsets[j + 1] - offsets[j]
        if count:
            memcopy(_index(target, "addresses"), int(flat_address) + 4 * offsets[j], 4 * count)


class TrainerOwnerEntries:
    """The trainer owner entries of the GPU byte LM binding
    (`byte_lm_trainer_*`; training/byte_lm_session_ops.mojo `byte_owner_*`),
    composed from a binding's own single-device entries in the same order:
    the ids admitted, the session opened from the host state when it is not
    open, the step or evaluation, the gradient export split by name, and the
    rollback and lost-session verdict after a failure. The CPU adapter below
    serves them this way; it holds no arithmetic."""

    def _owner_session_open(self, session):
        return bool(int(self.byte_lm_session_info(session)[3]))

    def _owner_lose(self, session):
        session.usable = False

    def _owner_mark(self, session, committed):
        """`byte_owner_mark`: lost when not usable at the committed step."""
        info = self.byte_lm_session_info(session)
        if int(info[3]) and (not int(info[2]) or int(info[0]) != committed):
            self._owner_lose(session)

    def _owner_recover(self, session, committed):
        """`byte_owner_recover`: roll back, then judge the session."""
        try:
            if not self._owner_session_open(session):
                return
            self.byte_lm_session_rollback(session)
        except Exception:
            self._owner_lose(session)
            return
        self._owner_mark(session, committed)

    def byte_lm_trainer_step(self, session, addresses, params, shape):
        cfg = _config(shape)
        a = [_index(x, "addresses") for x in addresses]
        train = _index(params[0], "action") == 1
        committed = _index(params[1], "completed")
        if len(a) < 8:
            raise ValueError("byte LM trainer: expected at least 8 addresses")
        full = a[7] != 0
        if full and not train:
            raise ValueError("byte LM trainer: an evaluation returns no gradient")
        if len(a) != (8 + cfg.n_tensors if full else 8):
            raise ValueError("byte LM trainer: expected one named gradient output per parameter tensor")
        if train and committed >= _MAX_COMPLETED:
            raise ValueError("byte LM: completed step outside admitted bound")
        if not train and a[6] != 0:
            raise ValueError("byte LM trainer: an evaluation requires a null flags output")
        _validate_tokens(_read_i32(a[0], cfg.batch * (cfg.length + 1)), cfg)
        if not self._owner_session_open(session):
            if not any(a[2:5]):
                raise ValueError("byte LM: session is not open")
            self.byte_lm_session_open(session, [a[2], a[3], a[4], a[1]], params, shape)
        try:
            if train:
                step = self.byte_lm_session_step(session, [a[0], a[1], a[5], a[6]], params, shape)
                if step != committed + 1:
                    raise RuntimeError("byte LM: successful result has wrong completed step")
                if full:
                    self.byte_lm_session_export_gradients(session, [a[7]], shape)
                    _scatter_named(a[7], a[8:], cfg)
            else:
                step = self.byte_lm_session_eval(session, [a[0], a[1], a[5]], params, shape)
                if step != committed:
                    raise RuntimeError("byte LM: successful result has wrong completed step")
        except BaseException:
            self._owner_recover(session, committed)
            raise
        return step

    def byte_lm_trainer_export_state(self, session, addresses, shape, completed, release):
        cfg = _config(shape)
        a = [_index(x, "addresses") for x in addresses]
        if len(a) != 5:
            raise ValueError("byte LM trainer: expected 5 addresses")
        nt = cfg.n_tensors
        expected = ctypes.string_at(a[0], 4 * nt)
        try:
            step = self.byte_lm_session_export_state(session, a[1:], shape)
            if step != completed:
                raise RuntimeError("Byte-LM state export returned a different completed step")
            if ctypes.string_at(a[4], 4 * nt) != expected:
                raise RuntimeError("Byte-LM exported flags differ from the admitted flags")
        except BaseException:
            if release:
                try:
                    self.byte_lm_session_close(session)
                except Exception:
                    pass
            raise
        if release:
            self.byte_lm_session_close(session)
        return step

    def byte_lm_trainer_export_gradients(self, session, addresses, shape, completed):
        cfg = _config(shape)
        a = [_index(x, "addresses") for x in addresses]
        if len(a) not in (1, 1 + cfg.n_tensors):
            raise ValueError("byte LM trainer: expected the flat gradient and, optionally, "
                             "one output per parameter tensor")
        step = self.byte_lm_session_export_gradients(session, a[:1], shape)
        if step != completed:
            raise RuntimeError("Byte-LM gradient export returned a different step")
        if len(a) > 1:
            _scatter_named(a[0], a[1:], cfg)
        return step

    def _owner_logits(self, session, addresses, dims, params, shape, next_bytes):
        cfg = _config(shape)
        a = [_index(x, "addresses") for x in addresses]
        if len(a) != 6:
            raise ValueError("byte LM trainer: expected 6 addresses")
        committed = _index(params[1], "completed")
        b, l = (_index(d, "dims") for d in dims)
        _validate_tokens(_read_i32(a[0], b * l), cfg)
        if not self._owner_session_open(session):
            if not any(a[3:6]):
                raise ValueError("byte LM: session is not open")
            self.byte_lm_session_open(session, [a[3], a[4], a[5], a[2]], params, shape)
        try:
            if next_bytes:
                return self.byte_lm_session_next_bytes(session, a[:2], dims, shape, committed)
            return self.byte_lm_session_logits(session, a[:2], dims, shape, committed)
        except BaseException:
            self._owner_mark(session, committed)
            raise

    # Present exactly when the session entry they compose is: a binding
    # without it reads as lacking the trainer entry (`getattr(..., None)`).
    @property
    def byte_lm_trainer_logits(self):
        if not callable(getattr(self, "byte_lm_session_logits", None)):
            raise AttributeError("byte_lm_trainer_logits needs byte_lm_session_logits")
        return lambda session, addresses, dims, params, shape: self._owner_logits(
            session, addresses, dims, params, shape, False)

    @property
    def byte_lm_trainer_next_bytes(self):
        if not callable(getattr(self, "byte_lm_session_next_bytes", None)):
            raise AttributeError("byte_lm_trainer_next_bytes needs byte_lm_session_next_bytes")
        return lambda session, addresses, dims, params, shape: self._owner_logits(
            session, addresses, dims, params, shape, True)

    def byte_lm_trainer_run(self, addresses, named, params, shape):
        cfg = _config(shape)
        named = [_index(x, "addresses") for x in named]
        if named and _index(params[0], "action") != 1:
            raise ValueError("byte LM trainer: an evaluation returns no gradient")
        run = getattr(self, "byte_lm_run_configured", None)
        result = run(addresses, params, shape) if callable(run) else self.byte_lm_run(addresses, params)
        if named:
            _scatter_named(addresses[8], named, cfg)
        return result


