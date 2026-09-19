# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SmallByteLanguageModelTrainer on a CPU-only install
(lane/cpu-training-embedding-ivf, 2026-09-15; the byte-lm and
byte-lm-resident lanes of tools/identity_break.py).

WHAT THIS IS. The single-device part of the GPU byte LM binding's ABI
(`bindings/_mojolearn_byte_lm.mojo`: `byte_lm_run`, `byte_lm_run_configured`,
`byte_lm_config_profile`, `byte_lm_logits` and the resident session entries),
served over the CPU byte LM binding `_mojolearn_byte_lm_host`, whose step
(`byte_lm_host_train_step`: forward, backward and AdamW through the host
oracles, DEVIATION 2680), loss (`byte_lm_host_loss`) and logits
(`byte_lm_host_logits`, reference path) are the arithmetic.
`_byte_lm_impl._load` returns `binding()` instead of the GPU binding when this
process loaded no GPU set (`_backend._CPU_ONLY`), so the trainer class, its
admission, its state dict and its checkpoint codec run unchanged. This file
holds no arithmetic.

WHY THE BITS ARE THE DEVICE'S. The host step reproduces the retained Apple,
NVIDIA and AMD step captures byte for byte (docs/BYTE_LM_CPU_TRAINING.md) and
the host logits the device logits (tools/byte_lm_gpu_logits_sweep.py). At the
lanes' own configuration (the default b2-l32 shape, lr 1e-3, betas (0.9,
0.999), eps 1e-8, weight decay 0.01, three steps) all 18 train, 36 infer and
model and 18 batch cells of the two lanes read IDENTICAL x4 against the
166-lane record's three GPU columns on the M4 (one core) before this file was
written, through a probe that replaced the same three doors.

WHAT IS RESTATED FROM THE DEVICE SIDE, and nothing more:
  the momentum flags   an AdamW step leaves `buf_initialized` as it was
                       (`training/checks/optimizer.mojo` sets it only under
                       SGD momentum), and the byte LM admits AdamW only, so a
                       step returns the flags it was handed; the probe's
                       model cells (the checkpoint bytes) agree.
  eval                 `action = 0` returns the loss and copies the state to
                       the outputs unchanged; `out_grad` must be 0.
  the session          `ByteLMSession` and `ByteTrainer`'s bookkeeping:
                       admission once, `completed_steps`, `grad_step`, the
                       shadow of the last step that reached its shadow point
                       (kept after a successful step, cleared when the next
                       step or evaluation starts, restored and cleared by
                       rollback), `usable` and `busy`, and `info` as
                       [completed, grad_step, usable, open]. A host step
                       writes fresh arrays, so a step that raises changes
                       nothing and has no shadow to restore.
  the refusals         the GPU binding's admission of addresses, action,
                       step bound, optimizer, flags, state and ids, in its
                       words where the Python class does not already refuse
                       first.

ABSENT, SO THEY REFUSE BY NAME (an AttributeError naming the entry and this
file, so `getattr(binding, name, None)` and `hasattr` still read absent): the
multi-GPU `byte_lm_parallel_*`, `byte_lm_offload_*` and `byte_lm_model_pool_*`
entries, `byte_lm_session_run`, the context keeper and fault injection
read-backs, and the attention and step glue arm read-backs
(`_binding_metadata` records None for both, as for a binding built before
them).
"""
import ctypes
from . import _portable_math as math
import operator
import struct

from . import _backend
from ._bufcheck import flat_view, memcopy
from ._buffer import addr, addr_ro, frombytes, zeros
from ._byte_lm_config import ByteLanguageModelConfig

HOST_BASENAME = "_mojolearn_byte_lm_host"
_IDENTICAL_CODE = 1
_MAX_COMPLETED = 999999
_LOGITS_MAX_BATCH = 1024
_LOGITS_MAX_CELLS = 268435456
_ADAMW = 2


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


def _read_f32(address, n):
    return frombytes(ctypes.string_at(int(address), 4 * n), "<f4", (n,))


def _read_i32(address, n):
    return frombytes(ctypes.string_at(int(address), 4 * n), "<i4", (n,))


def _words(array, code):
    return flat_view(array, code)


def _optimizer(params):
    """`OptimizerConfig` from params[2:12] and `byte_validate_optimizer`:
    AdamW, positive lr and eps, betas in [0, 1), nonnegative weight decay,
    no momentum, dampening, nesterov or clipping. Values narrowed to float32
    the way the binding narrows them."""
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", float(x)))[0]
    kind = _index(params[2], "kind")
    lr, b1, b2, eps, wd, mom, damp = (f32(params[i]) for i in (3, 4, 5, 6, 7, 8, 9))
    nesterov = _index(params[10], "nesterov")
    max_norm = f32(params[11])
    if nesterov not in (0, 1):
        raise ValueError("byte LM: action/nesterov must be 0 or 1")
    if (kind != _ADAMW or not lr > 0 or not eps > 0 or not 0 <= b1 < 1 or not 0 <= b2 < 1
            or not wd >= 0 or mom != 0 or damp != 0 or nesterov != 0 or max_norm != 0):
        raise ValueError("byte LM: the first profile admits positive-lr AdamW without clipping or SGD options")
    return (kind, lr, b1, b2, eps, wd)


def _validate_state(p, m, v, flags, completed, cfg):
    """`byte_validate_state`."""
    if not 0 <= completed < 1000000:
        raise ValueError("byte LM: completed step outside admitted bound")
    for name, arr in (("parameters", p), ("m", m), ("v", v)):
        for x in _words(arr, "f"):
            if not math.isfinite(x):
                raise ValueError(f"byte LM: nonfinite {name}")
    if any(x < 0 for x in _words(v, "f")):
        raise ValueError("byte LM: second moments must be nonnegative")
    if any(x not in (0, 1) for x in _words(flags, "i")):
        raise ValueError("byte LM: momentum flags must be exactly 0 or 1")


def _validate_tokens(ids, cfg):
    """`byte_validate_tokens`: every id in [0, vocab)."""
    for x in _words(ids, "i"):
        if not 0 <= x < cfg.vocab_size:
            raise ValueError("byte LM: token id outside [0, vocab)")


def _addresses(addresses, count, n_inputs, what):
    """`_read_addresses` and the nonzero half of `_validate_slot_table`."""
    if len(addresses) != count:
        raise ValueError(f"byte LM: {what} expects {count} addresses, got {len(addresses)}")
    out = [_index(a, "addresses") for a in addresses]
    if any(a == 0 for a in out[:n_inputs]):
        raise ValueError(f"byte LM: {what} has a null input address")
    return out


def _write_bits(address, value_f32):
    ctypes.memmove(int(address), struct.pack("<f", value_f32), 4)


def _loss_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


class _Session:
    """`ByteLMSession` holding a host `ByteTrainer`'s bookkeeping."""

    def __init__(self):
        self.state = None          # dict(p, m, v, flags, completed, grad, grad_step, cfg, opt, shadow, healthy)
        self.usable = True
        self.busy = False

    def close(self):
        self.state = None
        self.usable = False
        return 0


class _HostTrainerBinding:
    """The GPU byte LM binding's single-device ABI (module docstring)."""

    def __init__(self, host):
        self._host = host
        self.__file__ = host.__file__
        self.__name__ = "mojolearn._host._byte_lm_trainer"

    def __getattr__(self, name):
        raise AttributeError(
            f"mojolearn: no CPU implementation of _mojolearn_byte_lm.{name}; the CPU trainer "
            "(python/mojolearn/_byte_lm_trainer_host.py) serves the single-device entries only")

    # ---- the read-backs ---------------------------------------------------
    def byte_lm_numeric_mode(self):
        return int(self._host.byte_lm_host_numeric_mode())

    def byte_lm_vendor(self):
        return str(self._host.byte_lm_host_vendor())

    def byte_lm_profile(self):
        return str(self._host.byte_lm_host_profile(list(ByteLanguageModelConfig().native_shape)))

    def byte_lm_config_profile(self, shape):
        return str(self._host.byte_lm_host_profile(list(_config(shape).native_shape)))

    # ---- the stateless step ----------------------------------------------
    def _step(self, cfg, p_addr, m_addr, v_addr, ids_addr, grad_addr, out_p, out_m, out_v, opt, completed):
        """One host step on caller-owned addresses; returns the loss's float32."""
        _, lr, b1, b2, eps, wd = opt
        bits = self._host.byte_lm_host_train_step(
            [p_addr, m_addr, v_addr, ids_addr, grad_addr, out_p, out_m, out_v],
            list(cfg.native_shape), [lr, b1, b2, eps, wd], completed)
        return _loss_from_bits(bits)

    def _loss(self, cfg, p_addr, ids_addr):
        return _loss_from_bits(self._host.byte_lm_host_loss([p_addr, ids_addr], list(cfg.native_shape), 0, 1))

    def _run(self, addresses, params, cfg):
        """`_byte_lm_run` (module docstring)."""
        if len(addresses) != 11 or len(params) != 12:
            raise ValueError("byte LM: expected 11 addresses and 12 scalar parameters")
        n, nt = cfg.n_total, cfg.n_tensors
        action = _index(params[0], "action")
        completed = _index(params[1], "completed")
        if action not in (0, 1):
            raise ValueError("byte LM: action/nesterov must be 0 or 1")
        if completed < 0 or completed >= 1000000 or (action == 1 and completed >= _MAX_COMPLETED):
            raise ValueError("byte LM: completed step outside admitted bound")
        opt = _optimizer(params)
        a = _addresses(addresses, 11, 5, "byte_lm_run")
        if action == 0 and a[8] != 0:
            raise ValueError("byte LM: evaluation requires a null gradient address")
        if action == 1 and a[8] == 0:
            raise ValueError("byte LM: training requires a gradient address")
        if any(x == 0 for i, x in enumerate(a[5:]) if i != 3):
            raise ValueError("byte LM: null output address")
        p, m, v = _read_f32(a[0], n), _read_f32(a[1], n), _read_f32(a[2], n)
        flags = _read_i32(a[3], nt)
        ids = _read_i32(a[4], cfg.batch * (cfg.length + 1))
        _validate_state(p, m, v, flags, completed, cfg)
        _validate_tokens(ids, cfg)
        ids_addr = addr_ro(ids, name="ids")
        if action == 1:
            grad = zeros((n,), "<f4")
            out_p = zeros((n,), "<f4")
            out_m = zeros((n,), "<f4")
            out_v = zeros((n,), "<f4")
            loss = self._step(cfg, addr_ro(p, name="p"), addr_ro(m, name="m"), addr_ro(v, name="v"), ids_addr,
                              addr(grad, name="grad"), addr(out_p, name="p"), addr(out_m, name="m"),
                              addr(out_v, name="v"), opt, completed)
            if not math.isfinite(loss):
                raise ValueError("byte LM: nonfinite returned loss")
            memcopy(a[8], addr_ro(grad, name="grad"), 4 * n)
            memcopy(a[5], addr_ro(out_p, name="p"), 4 * n)
            memcopy(a[6], addr_ro(out_m, name="m"), 4 * n)
            memcopy(a[7], addr_ro(out_v, name="v"), 4 * n)
            result = completed + 1
        else:
            loss = self._loss(cfg, addr_ro(p, name="p"), ids_addr)
            if not math.isfinite(loss):
                raise ValueError("byte LM: nonfinite returned loss")
            memcopy(a[5], addr_ro(p, name="p"), 4 * n)
            memcopy(a[6], addr_ro(m, name="m"), 4 * n)
            memcopy(a[7], addr_ro(v, name="v"), 4 * n)
            result = completed
        memcopy(a[9], addr_ro(flags, name="flags"), 4 * nt)
        _write_bits(a[10], loss)
        return result

    def byte_lm_run(self, addresses, params):
        return self._run(addresses, params, ByteLanguageModelConfig())

    def byte_lm_run_configured(self, addresses, params, shape):
        return self._run(addresses, params, _config(shape))

    # ---- logits ------------------------------------------------------------
    def _dims(self, dims, cfg):
        if len(dims) != 2:
            raise ValueError("byte LM logits: expected dims [batch, length]")
        b, l = (_index(d, "dims") for d in dims)
        if b < 1 or b > _LOGITS_MAX_BATCH or l < 1 or l > cfg.length:
            raise ValueError(f"byte LM logits: batch in [1, {_LOGITS_MAX_BATCH}] and length in [1, {cfg.length}]")
        if b * l > _LOGITS_MAX_CELLS // cfg.vocab_size:
            raise ValueError("byte LM logits: batch * length * vocab exceeds the admitted span")
        return b, l

    def _logits(self, cfg, p_addr, ids_addr, out_addr, b, l):
        written = int(self._host.byte_lm_host_logits([p_addr, ids_addr, out_addr], [b, l],
                                                     list(cfg.native_shape), 0, 1))
        if written != b * l * cfg.vocab_size:
            raise ValueError("byte LM logits: wrong logits length")
        for x in _words(frombytes(ctypes.string_at(out_addr, 4 * written), "<f4", (written,)), "f"):
            if not math.isfinite(x):
                raise ValueError("byte LM logits: nonfinite logit")
        return written

    def byte_lm_logits(self, addresses, dims, shape):
        cfg = _config(shape)
        b, l = self._dims(dims, cfg)
        a = _addresses(addresses, 3, 3, "byte_lm_logits")
        p = _read_f32(a[0], cfg.n_total)
        ids = _read_i32(a[1], b * l)
        _validate_tokens(ids, cfg)
        for x in _words(p, "f"):
            if not math.isfinite(x):
                raise ValueError("byte LM logits: nonfinite parameters")
        return self._logits(cfg, addr_ro(p, name="p"), addr_ro(ids, name="ids"), a[2], b, l)

    # ---- the resident session ---------------------------------------------
    @staticmethod
    def _owner(session):
        if not isinstance(session, _Session):
            raise TypeError("byte LM: not a session created by byte_lm_session_create")
        return session

    @staticmethod
    def _require_open(owner):
        if owner.state is None:
            raise ValueError("byte LM: session is not open")
        if owner.busy:
            raise ValueError("byte LM: session is busy")
        if not owner.state["healthy"]:
            raise ValueError("byte LM: session lost; restore a retained export")

    def byte_lm_session_create(self):
        return _Session()

    def byte_lm_session_close(self, session):
        return self._owner(session).close()

    def byte_lm_session_open(self, session, addresses, params, shape):
        cfg = _config(shape)
        owner = self._owner(session)
        n, nt = cfg.n_total, cfg.n_tensors
        completed = _index(params[1], "completed")
        opt = _optimizer(params)
        a = _addresses(addresses, 4, 4, "byte_lm_session_open")
        if owner.busy:
            raise ValueError("byte LM: session is busy")
        if owner.state is not None:
            raise ValueError("byte LM: session is already open; close it first")
        p, m, v = _read_f32(a[0], n), _read_f32(a[1], n), _read_f32(a[2], n)
        flags = _read_i32(a[3], nt)
        _validate_state(p, m, v, flags, completed, cfg)
        owner.state = dict(cfg=cfg, opt=opt, p=p, m=m, v=v, flags=flags, completed=completed,
                           grad=None, grad_step=-1, shadow=None, healthy=True)
        owner.usable = True
        return completed

    def _admit(self, st, params, flags, cfg):
        """`_admit_session_scalars`: the caller's step, optimizer, flags and
        shape must be the session's."""
        if cfg.native_shape != st["cfg"].native_shape:
            raise ValueError("byte LM: resident model shape mismatch")
        if _index(params[1], "completed") != st["completed"]:
            raise ValueError("byte LM: resident completed-step mismatch")
        if _optimizer(params) != st["opt"]:
            raise ValueError("byte LM: resident optimizer mismatch")
        if flags.tobytes() != st["flags"].tobytes():
            raise ValueError("byte LM: resident momentum flags mismatch")

    def byte_lm_session_step(self, session, addresses, params, shape):
        cfg = _config(shape)
        owner = self._owner(session)
        if _index(params[0], "action") != 1:
            raise ValueError("byte LM: session step requires action 1")
        completed = _index(params[1], "completed")
        if completed >= _MAX_COMPLETED:
            raise ValueError("byte LM: completed step outside admitted bound")
        a = _addresses(addresses, 4, 2, "byte_lm_session_step")
        ids = _read_i32(a[0], cfg.batch * (cfg.length + 1))
        flags = _read_i32(a[1], cfg.n_tensors)
        _validate_tokens(ids, cfg)
        self._require_open(owner)
        st = owner.state
        self._admit(st, params, flags, cfg)
        n = cfg.n_total
        st["shadow"] = None
        st["grad_step"] = -1
        grad = zeros((n,), "<f4")
        out_p = zeros((n,), "<f4")
        out_m = zeros((n,), "<f4")
        out_v = zeros((n,), "<f4")
        owner.busy = True
        try:
            loss = self._step(cfg, addr_ro(st["p"], name="p"), addr_ro(st["m"], name="m"),
                              addr_ro(st["v"], name="v"), addr_ro(ids, name="ids"), addr(grad, name="grad"),
                              addr(out_p, name="p"), addr(out_m, name="m"), addr(out_v, name="v"),
                              st["opt"], st["completed"])
            if not math.isfinite(loss):
                raise ValueError("byte LM: nonfinite returned loss")
        finally:
            owner.busy = False
        st["shadow"] = (st["p"], st["m"], st["v"], st["flags"], st["completed"])
        st.update(p=out_p, m=out_m, v=out_v, grad=grad, completed=st["completed"] + 1)
        st["grad_step"] = st["completed"]
        memcopy(a[3], addr_ro(st["flags"], name="flags"), 4 * cfg.n_tensors)
        _write_bits(a[2], loss)
        return st["completed"]

    def byte_lm_session_eval(self, session, addresses, params, shape):
        cfg = _config(shape)
        owner = self._owner(session)
        if _index(params[0], "action") != 0:
            raise ValueError("byte LM: session eval requires action 0")
        a = _addresses(addresses, 3, 2, "byte_lm_session_eval")
        ids = _read_i32(a[0], cfg.batch * (cfg.length + 1))
        flags = _read_i32(a[1], cfg.n_tensors)
        _validate_tokens(ids, cfg)
        self._require_open(owner)
        st = owner.state
        self._admit(st, params, flags, cfg)
        st["shadow"] = None
        loss = self._loss(cfg, addr_ro(st["p"], name="p"), addr_ro(ids, name="ids"))
        if not math.isfinite(loss):
            raise ValueError("byte LM: nonfinite returned loss")
        _write_bits(a[2], loss)
        return st["completed"]

    def byte_lm_session_export_state(self, session, addresses, shape):
        cfg = _config(shape)
        owner = self._owner(session)
        a = _addresses(addresses, 4, 0, "byte_lm_session_export_state")
        if any(x == 0 for x in a):
            raise ValueError("byte LM: null output address")
        self._require_open(owner)
        st = owner.state
        if cfg.native_shape != st["cfg"].native_shape:
            raise ValueError("byte LM: resident model shape mismatch")
        _validate_state(st["p"], st["m"], st["v"], st["flags"], st["completed"], cfg)
        n = cfg.n_total
        memcopy(a[0], addr_ro(st["p"], name="p"), 4 * n)
        memcopy(a[1], addr_ro(st["m"], name="m"), 4 * n)
        memcopy(a[2], addr_ro(st["v"], name="v"), 4 * n)
        memcopy(a[3], addr_ro(st["flags"], name="flags"), 4 * cfg.n_tensors)
        return st["completed"]

    def byte_lm_session_export_gradients(self, session, addresses, shape):
        cfg = _config(shape)
        owner = self._owner(session)
        a = _addresses(addresses, 1, 0, "byte_lm_session_export_gradients")
        if a[0] == 0:
            raise ValueError("byte LM: null output address")
        self._require_open(owner)
        st = owner.state
        if cfg.native_shape != st["cfg"].native_shape:
            raise ValueError("byte LM: resident model shape mismatch")
        if st["grad_step"] < 0 or st["grad_step"] != st["completed"]:
            raise ValueError("byte LM: no gradient to export; complete a step first")
        for x in _words(st["grad"], "f"):
            if not math.isfinite(x):
                raise ValueError("byte LM: nonfinite returned gradient")
        memcopy(a[0], addr_ro(st["grad"], name="grad"), 4 * cfg.n_total)
        return st["grad_step"]

    def byte_lm_session_logits(self, session, addresses, dims, shape, completed):
        cfg = _config(shape)
        owner = self._owner(session)
        b, l = self._dims(dims, cfg)
        claimed = _index(completed, "completed")
        if claimed < 0 or claimed >= 1000000:
            raise ValueError("byte LM: completed step outside admitted bound")
        a = _addresses(addresses, 2, 1, "byte_lm_session_logits")
        ids = _read_i32(a[0], b * l)
        _validate_tokens(ids, cfg)
        self._require_open(owner)
        st = owner.state
        if cfg.native_shape != st["cfg"].native_shape:
            raise ValueError("byte LM: resident model shape mismatch")
        if st["completed"] != claimed:
            raise ValueError("byte LM: resident completed-step mismatch")
        return self._logits(cfg, addr_ro(st["p"], name="p"), addr_ro(ids, name="ids"), a[1], b, l)

    def byte_lm_session_rollback(self, session):
        """`byte_rollback`: restore the last step's shadow if one is held."""
        owner = self._owner(session)
        if owner.state is None:
            raise ValueError("byte LM: session is not open")
        st = owner.state
        if st["shadow"] is not None:
            p, m, v, flags, step = st["shadow"]
            st.update(p=p, m=m, v=v, flags=flags, completed=step, shadow=None, grad=None, grad_step=-1)
        return st["completed"]

    def byte_lm_session_info(self, session):
        owner = self._owner(session)
        if owner.state is None:
            return [-1, -1, 1 if owner.usable else 0, 0]
        st = owner.state
        return [st["completed"], st["grad_step"], 1 if owner.usable else 0, 1]


_BINDING = None


def is_cpu_trainer_binding(value):
    """Whether `value` is this adapter, the one binding `_byte_lm_impl._load`
    admits under the "cpu" vendor."""
    return isinstance(value, _HostTrainerBinding)


def binding():
    """The adapter over the CPU byte LM binding, loaded (and read back as the
    CPU column, IDENTICAL, not a sabotage build outside the gate) by
    `_backend.load_host_module`, so MOJOLEARN_HOST_DIR and
    MOJOLEARN_HOST_ALLOW_SABOTAGE apply as for every routed family."""
    global _BINDING
    host = _backend.load_host_module(HOST_BASENAME)
    if _BINDING is None or _BINDING._host is not host:
        missing = [name for name in ("byte_lm_host_train_step", "byte_lm_host_loss", "byte_lm_host_logits",
                                     "byte_lm_host_profile") if not callable(getattr(host, name, None))]
        if missing:
            raise ImportError(f"mojolearn: {host.__file__} lacks {', '.join(missing)}; "
                              "rebuild bindings/build_byte_lm_host.sh")
        _BINDING = _HostTrainerBinding(host)
    return _BINDING
