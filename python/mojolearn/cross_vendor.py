# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Live cross-vendor data-parallel training. EXPERIMENTAL.

One language model trained at the same time by workers on GPUs from different
vendors (Apple, NVIDIA, AMD), with every replica bit-identical after every
step, and the step itself identical to a one-GPU `train_step` over the same
logical shards.

    # the coordinator: any machine, no GPU needed
    python -m mojolearn.cross_vendor coordinator --port 7777 --workers 2 --shards 4 --steps 100

    # each worker, in Python, with its own GPU
    from mojolearn.cross_vendor import Worker
    Worker(state, shards=[0, 1], batches=lambda step, shard: ids,
           address=("coordinator-host", 7777), name="my-mac").run()

WHY THE BITS AGREE. A step is K logical shards (K is part of the recipe, not
of the hardware). Each worker computes the gradient of the shards it owns
with the same IDENTICAL kernels a one-GPU step uses, so a shard's gradient
does not depend on the GPU or its vendor. The coordinator combines them with
the fixed left fold the device reduction uses (`ordered_fold`): total = g0,
then total = ftz(ftz(total) + ftz(g_k)) in shard order. Every worker applies
that same total with the same elementwise AdamW. Nothing chooses an
arithmetic order at run time.

WHAT IS CHECKED, EVERY STEP. Every worker reports the sha256 of its complete
state (parameters, m, v, flags) after the update; the coordinator refuses to
go on, naming the workers, if any two differ. Workers must also start from
the same state hash and together own every shard exactly once.

TWO WAYS TO MOVE THE BYTES. The GATHERED protocol (the default) has every
worker send every shard's whole gradient (4 bytes per parameter) to the
coordinator, which folds them all and sends the total back: K + W gradients
on the wire per step, fine for a small model on a local network, 41 GB a
step at 162M parameters and K = 64. The CHAINED protocol (`chained=True`)
uses the fact that the fold is a LEFT fold: each worker owns a CONTIGUOUS
block of shards in shard order; the first block's owner folds its own
shards and sends that ONE prefix; the next owner continues the fold onto
the prefix with its own shards, one at a time in order, and sends the
result on; the last owner's result is the total, which the coordinator
sends to every other worker. Bit for bit the flat fold a single GPU
computes, and W + (W - 1) gradients on the wire per step instead of K + W:
with two workers and the coordinator on the first owner's box, two
gradients cross the wide-area link. A worker holds its block's gradients
in host memory until its turn (`shards x 4 bytes x n_total`).

No authentication or encryption: run it on a trusted network or through an
ssh tunnel. Each worker drives one device with a replicated optimizer
(`ParallelByteLanguageModelTrainer(state, devices=(d,), pool_optimizer=False)`).
"""
import argparse
import array
import hashlib
import json
import socket
import struct
import sys
import threading
import time

try:
    from ._buffer import flat_bytes
except (ImportError, AttributeError):  # loaded by file path outside the package (tools/tests do this
    # for the host fold, which needs only the standard library): a memoryview
    # is enough there, since the package's Array never reaches such a caller
    def flat_bytes(obj, *, name="array"):
        return memoryview(obj).cast("B")

PROTOCOL = "mojolearn.cross-vendor.v1"
_TINY = 2.0 ** -126  # smallest normal float32

__all__ = ["ordered_fold", "fold_pair", "state_hash", "Coordinator", "Worker", "CrossVendorMismatch", "PROTOCOL"]


class CrossVendorMismatch(RuntimeError):
    """Two workers disagree, or the group is not a valid split of one step."""


from ._backend import load_host_module as _load_host_module
from ._array import Array as _Array
from ._buffer import _native as _buffer_native
_native_cross_vendor = _load_host_module("_mojolearn_tokenizer_host").cross_vendor_call
_native_cross_vendor("initialize", (_Array, CrossVendorMismatch, _buffer_native("fold_pair_f32")))


# ------------------------------------------------------------ the fold


def _f32(payload):
    return _native_cross_vendor("f32", (payload,))


def fold_pair(total, shard):
    """Add two float32 buffers with the fixed FTZ reduction contract."""
    return _native_cross_vendor("pair", (total, shard))


def ordered_fold(gradients, *, prefix=None):
    """Fold float32 shard buffers in logical order, optionally after a prefix."""
    return _native_cross_vendor("fold", (gradients, prefix))


def state_hash(state):
    """Return SHA256 over parameters, m, v and flags, in that order."""
    return _native_cross_vendor("state_hash", (state,))


def trainer_state_hash(trainer):
    """`state_hash` of a trainer's current replica, through `export_raw`
    when the trainer has it (no per-element admission pass, which costs
    about 16 s at 162M parameters) and `state_dict` otherwise."""
    raw = getattr(trainer, "export_raw", None)
    return state_hash(raw() if callable(raw) else trainer.state_dict())


# ------------------------------------------------------------ framing
def _send(sock, header, payload=b""):
    return _native_cross_vendor("send", (sock, header, payload))


def _recv(sock, max_payload):
    return _native_cross_vendor("receive", (sock, max_payload))


def _recv_exact(sock, n):
    return _native_cross_vendor("receive_exact", (sock, n))


_cross_vendor_errors = {"mismatch": CrossVendorMismatch}


class _NativeReady:
    def __init__(self):
        return _native_cross_vendor("event_init", (self,))

    def set(self):
        return _native_cross_vendor("event_set", (self,))

    def clear(self):
        return _native_cross_vendor("event_clear", (self,))

    def is_set(self):
        return _native_cross_vendor("event_is_set", (self,))

    def wait(self, timeout=None):
        return _native_cross_vendor("event_wait", (self, timeout))


# ------------------------------------------------------------ coordinator
class Coordinator:
    """Accepts `workers` connections, then drives `steps` lockstep steps.

    `run()` returns one row per step: the step number, every shard's loss,
    the agreed state hash and each worker's hash. `on_step(row)` is called
    after each agreed step. Refuses, naming what differs, on a bad split, a
    different starting state or a disagreement after any step."""

    def __init__(self, *, port, workers, logical_shards, steps, host="0.0.0.0",
                 on_step=None, timeout=3600.0, accept_timeout=3600.0, max_gradient_bytes=1 << 31,
                 chained=False):
        return _native_cross_vendor("coordinator_init", (self, port, workers, logical_shards, steps, host, on_step, timeout, accept_timeout, max_gradient_bytes, chained, _NativeReady), _cross_vendor_errors)

    def _refuse(self, reason):
        return _native_cross_vendor("coordinator_refuse", (self, reason), _cross_vendor_errors)

    def run(self):
        return _native_cross_vendor("coordinator_run", (self,), _cross_vendor_errors)

    def _drive(self):
        return _native_cross_vendor("coordinator_drive", (self,), _cross_vendor_errors)


# ------------------------------------------------------------ worker
class Worker:
    """One device's part of every step.

    `trainer` is a `ParallelByteLanguageModelTrainer(state, devices=(d,),
    pool_optimizer=False)`, or pass `state` and `device` and one is built.
    `batches(step, shard)` returns that shard's int32 [batch, length + 1]
    tokens; it must be the same function on every worker."""

    def __init__(self, state=None, *, shards, batches, address, name=None, device=0,
                 trainer=None, connect_timeout=600.0, timeout=3600.0, chained=False, lr_for_step=None,
                 on_commit=None):
        if trainer is None:
            from .parallel_training import ParallelByteLanguageModelTrainer
            trainer = ParallelByteLanguageModelTrainer(state, devices=(device,), logical_shards=1,
                                                       pool_optimizer=False)
        self.trainer, self.shards, self.batches = trainer, sorted(int(k) for k in shards), batches  # glue: validates the shard ids argument
        self.address, self.connect_timeout, self.timeout = tuple(address), connect_timeout, timeout
        self.name = name or socket.gethostname()
        self.chained = bool(chained)
        #: optional `lr_for_step(step) -> float`: the per-step learning rate
        #: (a recipe's table), applied through `trainer.set_lr` before the update
        self.lr_for_step = lr_for_step
        #: optional `on_commit(step_completed, row)` after each applied step, with
        #: the state hash, every shard's loss and the total's hash; the worker is
        #: idle inside the call, so its trainer may be exported or checkpointed
        self.on_commit = on_commit

    def _connect(self):
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                sock = socket.create_connection(self.address, timeout=30)
                sock.settimeout(self.timeout)
                return sock
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2)

    def run(self):
        """Serve until the coordinator says done; returns the steps committed."""
        from . import vendor
        tr = self.trainer
        raw = getattr(tr, "export_raw", None)
        state = raw() if callable(raw) else tr.state_dict()
        n_total = len(flat_bytes(state["parameters"], name="parameters")) // 4
        hello = dict(protocol=PROTOCOL, name=self.name, vendor=vendor(), shards=self.shards,
                     completed=tr.step_, state=state_hash(state), n_total=n_total, chained=self.chained)
        # A tunnel accepts the TCP connection before the coordinator behind it
        # listens (ssh -L answers locally, then closes when the far port
        # refuses), so a closed connection right after the hello is retried
        # like a refused one, until connect_timeout.
        deadline = time.monotonic() + self.connect_timeout
        while True:
            sock = self._connect()
            try:
                _send(sock, hello)
                head, payload = _recv(sock, n_total * 4)
                break
            except (ConnectionError, OSError):
                sock.close()
                if time.monotonic() > deadline:
                    raise
                time.sleep(5)
        committed = 0
        held, total, on_device = [], None, False  # chained: this block's gradients until its turn, then the fold
        first_message = True
        try:
            while True:
                if first_message:
                    first_message = False
                else:
                    head, payload = _recv(sock, n_total * 4)
                cmd = head.get("cmd")
                if cmd == "step":
                    if head["step"] != tr.step_:
                        raise CrossVendorMismatch("%s is at step %d, asked for %d" % (self.name, tr.step_, head["step"]))
                    if bool(head.get("chained")) != self.chained:
                        raise CrossVendorMismatch("%s: coordinator and worker disagree on the protocol" % self.name)
                    if self.lr_for_step is not None:
                        tr.set_lr(self.lr_for_step(head["step"]))
                    losses, parts = [], []
                    device_fold = self.chained and bool(getattr(tr, "has_device_fold", False))
                    if device_fold and head.get("first"):
                        # the first block: fold on the device as the shards are
                        # computed; nothing is downloaded until the prefix goes out
                        tr.fold_reset(None)
                        for k in self.shards:  # glue: one device gradient call per owned shard
                            losses.append(tr.shard_gradient_fold(self.batches(head["step"], k)))
                        held, total, on_device = [], None, True
                    else:
                        for k in self.shards:  # glue: one device gradient call per owned shard
                            loss, g = tr.shard_gradient(self.batches(head["step"], k))
                            losses.append(loss)
                            parts.append(flat_bytes(g, name="shard gradient").tobytes())
                        on_device = False
                    if self.chained:
                        if not on_device:
                            held, total = parts, None
                        _send(sock, {"step": head["step"], "shards": self.shards, "losses": losses})
                    else:
                        _send(sock, {"step": head["step"], "shards": self.shards, "losses": losses}, b"".join(parts))
                elif cmd == "fold":
                    if not self.chained or not (held or on_device):
                        raise CrossVendorMismatch("%s was asked to fold with nothing held" % self.name)
                    if on_device:
                        total = tr.fold_export()
                    elif bool(getattr(tr, "has_device_fold", False)):
                        # a prefix arrived: continue the fold on the device from it
                        tr.fold_reset(payload if payload else None)
                        for g in held:  # glue: one device fold call per held shard
                            tr.fold_add(g)
                        total = tr.fold_export()
                    else:
                        total = ordered_fold(held, prefix=payload if payload else None)
                    held, on_device = [], False
                    _send(sock, {"step": head["step"]}, total)
                elif cmd == "apply":
                    if payload:
                        total = payload
                    if total is None or len(total) != n_total * 4:
                        raise CrossVendorMismatch("summed gradient has the wrong size")
                    want = head.get("total_sha256")
                    if want is not None and hashlib.sha256(total).hexdigest() != want:
                        raise CrossVendorMismatch("%s holds a total whose hash is not the coordinator's" % self.name)
                    tr.apply_gradient(array.array("f", total))
                    applied, total = total, None
                    committed += 1
                    digest = trainer_state_hash(tr)
                    if self.on_commit is not None:
                        self.on_commit(tr.step_, dict(step=tr.step_, state=digest, total_sha256=want,
                                                      losses=head.get("losses"), total_bytes=applied))
                    _send(sock, {"completed": tr.step_, "state": digest})
                elif cmd == "done":
                    return committed
                elif cmd == "refuse":
                    raise CrossVendorMismatch("coordinator refused: " + head.get("reason", ""))
                else:
                    raise ConnectionError("cross_vendor: unknown command %r" % cmd)
        finally:
            sock.close()


# ------------------------------------------------------------ command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mojolearn.cross_vendor",
                                 description="Coordinator for live cross-vendor training (EXPERIMENTAL).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coordinator")
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=7777)
    c.add_argument("--workers", type=int, required=True)
    c.add_argument("--shards", type=int, required=True, help="logical shards per step, K")
    c.add_argument("--steps", type=int, required=True, help="train until this many steps are complete")
    c.add_argument("--log", help="append one JSON line per agreed step")
    c.add_argument("--chained", action="store_true",
                   help="the chained fold: contiguous blocks, one gradient per worker on the wire")
    args = ap.parse_args(argv)

    def on_step(row):
        line = json.dumps(row, sort_keys=True)
        print("step %d state %s %s" % (row["step"], row["state"][:16], sorted(row["vendors"].items())), flush=True)  # glue: vendor names in the log line
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(line + "\n")

    rows = Coordinator(host=args.host, port=args.port, workers=args.workers, logical_shards=args.shards,
                       steps=args.steps, on_step=on_step, chained=args.chained).run()
    print("AGREED on %d steps across %d workers" % (len(rows), args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
