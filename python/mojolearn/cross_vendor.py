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

PROTOCOL = "mojolearn.cross-vendor.v1"
_TINY = 2.0 ** -126  # smallest normal float32

__all__ = ["ordered_fold", "fold_pair", "state_hash", "Coordinator", "Worker", "CrossVendorMismatch", "PROTOCOL"]


class CrossVendorMismatch(RuntimeError):
    """Two workers disagree, or the group is not a valid split of one step."""


# ------------------------------------------------------------ the fold
_EXP, _MAN, _SIGN = 0x7F800000, 0x007FFFFF, 0x80000000


def _f32(payload):
    return memoryview(payload).cast("B").cast("f")


def _flush(bits, i):
    """checks/numerics.mojo `ftz` on element i of a uint32 view: a subnormal
    becomes a zero of the same sign."""
    b = bits[i]
    if not b & _EXP and b & _MAN:
        bits[i] = b & _SIGN


def fold_pair(total, shard, *, numpy=None):
    """One step of the device reduction on the host:
    ftz(ftz(total) + ftz(shard)) elementwise, one float32 rounding per add
    (training/byte_lm_parallel.mojo `_ordered_add_kernel`, fma(1, a, b)).
    Both are float32 buffers of the same length; returns bytes.

    Exact in pure Python: a float64 sum of two float32 values, rounded once
    to float32 on the store, is the float32 sum (53 >= 2*24 + 2). With
    NumPy installed the same arithmetic runs vectorized (a float32 add IS
    one rounding, and ftz is bit masking); the two spellings are held equal
    by `test_cross_vendor.py`. `numpy=False` forces the pure path."""
    t, g = _f32(total), _f32(shard)
    if len(t) != len(g):
        raise ValueError("fold_pair: gradients differ in length")
    np = None
    if numpy is not False:
        try:
            import numpy as np
        except ImportError:
            np = None
    if np is not None:
        # In place, over views of the callers' buffers: one owned copy of
        # each operand's bits, flushed where subnormal, one float32 add
        # into the first, one flush of the sum. No bytes() round trips and
        # no np.where temporaries: at 162M elements each of those was a
        # 649 MB allocation, and a 44-shard fold took 184 s on an H100 pod's
        # host (bench/results/lm_t1_2026-09-22/live).
        tb = np.array(np.frombuffer(t.cast("B"), dtype=np.uint32), copy=True)
        gb = np.array(np.frombuffer(g.cast("B"), dtype=np.uint32), copy=True)

        def ftz_(b):
            sub = (b & _EXP) == 0
            sub &= (b & _MAN) != 0
            b[sub] &= np.uint32(_SIGN)
        ftz_(tb)
        ftz_(gb)
        tf = tb.view(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            np.add(tf, gb.view(np.float32), out=tf)
        ftz_(tb)
        return tb.tobytes()
    n = len(t)
    out = array.array("f", t)
    sh = array.array("f", g)
    obits = memoryview(out).cast("B").cast("I")
    sbits = memoryview(sh).cast("B").cast("I")
    for i in range(n):
        _flush(obits, i)
        _flush(sbits, i)
        out[i] = out[i] + sh[i]
        _flush(obits, i)
    return out.tobytes()


def ordered_fold(gradients, *, prefix=None, numpy=None):
    """The device reduction, on the host: copy g[0] (or start from `prefix`,
    an already folded run of the shards BEFORE these), then
    total = fold_pair(total, g[k]) in shard order. `gradients` is a
    sequence of float32 buffers in shard order; returns bytes. A left fold,
    so folding a prefix and then the rest equals folding everything at
    once; that is what lets the chained protocol move one gradient per
    worker."""
    gradients = list(gradients)
    if prefix is None:
        if not gradients:
            raise ValueError("ordered_fold needs at least one gradient")
        total = bytes(_f32(gradients[0]).cast("B"))
        rest = gradients[1:]
    else:
        total = bytes(_f32(prefix).cast("B"))
        rest = gradients
    n = len(_f32(total))
    if any(len(_f32(g)) != n for g in rest):
        raise ValueError("ordered_fold: gradients differ in length")
    for g in rest:
        total = fold_pair(total, g, numpy=numpy)
    return total


def state_hash(state):
    """sha256 over parameters, m, v and flags, in that order."""
    h = hashlib.sha256()
    for key in ("parameters", "m", "v", "flags"):
        h.update(memoryview(state[key]).cast("B"))
    return h.hexdigest()


def trainer_state_hash(trainer):
    """`state_hash` of a trainer's current replica, through `export_raw`
    when the trainer has it (no per-element admission pass, which costs
    about 16 s at 162M parameters) and `state_dict` otherwise."""
    raw = getattr(trainer, "export_raw", None)
    return state_hash(raw() if callable(raw) else trainer.state_dict())


# ------------------------------------------------------------ framing
def _send(sock, header, payload=b""):
    header = dict(header, nbytes=len(payload))
    raw = json.dumps(header, sort_keys=True).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("cross_vendor: peer closed the connection")
        got += k
    return bytes(buf)


def _recv(sock, max_payload):
    (size,) = struct.unpack(">I", _recv_exact(sock, 4))
    if size > 1 << 20:
        raise ConnectionError("cross_vendor: oversized header")
    header = json.loads(_recv_exact(sock, size))
    nbytes = int(header.get("nbytes", 0))
    if not 0 <= nbytes <= max_payload:
        raise ConnectionError("cross_vendor: payload of %d bytes refused" % nbytes)
    return header, _recv_exact(sock, nbytes) if nbytes else b""


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
        if workers < 1 or logical_shards < workers:
            raise ValueError("need 1 <= workers <= logical_shards")
        self.host, self.port, self.n_workers = host, port, workers
        self.K, self.steps, self.on_step = logical_shards, steps, on_step
        self.timeout, self.accept_timeout, self.max_bytes = timeout, accept_timeout, max_gradient_bytes
        self.chained = bool(chained)
        self.peers = []
        self.bound_port = None
        self.ready = threading.Event()  # set once listening; `bound_port` is then the real port

    def _refuse(self, reason):
        for p in self.peers:
            try:
                _send(p["sock"], {"cmd": "refuse", "reason": reason})
            except OSError:
                pass
        raise CrossVendorMismatch(reason)

    def run(self):
        srv = socket.create_server((self.host, self.port), reuse_port=False)
        srv.settimeout(self.accept_timeout)
        self.bound_port = srv.getsockname()[1]
        self.ready.set()
        try:
            while len(self.peers) < self.n_workers:
                sock, where = srv.accept()
                sock.settimeout(self.timeout)
                hello, _ = _recv(sock, 0)
                self.peers.append(dict(sock=sock, where=where, **hello))
            return self._drive()
        finally:
            for p in self.peers:
                p["sock"].close()
            srv.close()

    def _drive(self):
        names = [p["name"] for p in self.peers]
        if any(p.get("protocol") != PROTOCOL for p in self.peers):
            self._refuse("protocol mismatch: " + str({p["name"]: p.get("protocol") for p in self.peers}))
        owned = sorted(k for p in self.peers for k in p["shards"])
        if owned != list(range(self.K)):
            self._refuse("workers must own shards 0..%d exactly once; got %s"
                         % (self.K - 1, {p["name"]: p["shards"] for p in self.peers}))
        for key in ("state", "completed", "n_total"):
            if len({json.dumps(p[key]) for p in self.peers}) != 1:
                self._refuse("workers start from different %s: %s" % (key, {p["name"]: p[key] for p in self.peers}))
        n_total = int(self.peers[0]["n_total"])
        if self.chained:
            # Contiguous blocks in shard order, so that each worker's fold of
            # its own shards is a run of the flat left fold.
            self.peers.sort(key=lambda p: p["shards"][0])
            for p in self.peers:
                if p["shards"] != list(range(p["shards"][0], p["shards"][0] + len(p["shards"]))):
                    self._refuse("chained fold: %s must own a contiguous block, got %s" % (p["name"], p["shards"]))
            if any(not p.get("chained") for p in self.peers):
                self._refuse("chained fold: every worker must be started with chained=True")
        rows = []
        for step in range(int(self.peers[0]["completed"]), self.steps):
            timing = {"step_sent": time.monotonic()}
            for i, p in enumerate(self.peers):
                # `first`: this worker's block opens the fold (no prefix will come)
                _send(p["sock"], {"cmd": "step", "step": step, "chained": self.chained, "first": self.chained and i == 0})
            grads, losses = {}, {}
            if self.chained:
                for p in self.peers:
                    head, _ = _recv(p["sock"], 0)
                    if head.get("step") != step or head.get("shards") != p["shards"]:
                        self._refuse("%s answered the wrong step or shards" % p["name"])
                    for i, k in enumerate(p["shards"]):
                        losses[k] = head["losses"][i]
                    timing["gradients_" + p["name"]] = round(time.monotonic() - timing["step_sent"], 3)
                prefix = b""
                for p in self.peers:
                    t0 = time.monotonic()
                    _send(p["sock"], {"cmd": "fold", "step": step}, prefix)
                    head, prefix = _recv(p["sock"], n_total * 4)
                    if head.get("step") != step or len(prefix) != n_total * 4:
                        self._refuse("%s returned a wrong fold" % p["name"])
                    # the prefix out, the worker's fold, the result back: one round trip per worker
                    timing["fold_" + p["name"]] = round(time.monotonic() - t0, 3)
                total = prefix
                last = self.peers[-1]["name"]
                all_losses = [losses[k] for k in range(self.K)]
                for p in self.peers:
                    # the last folder already holds the total; send it the hash only
                    _send(p["sock"], {"cmd": "apply", "step": step, "total_sha256": hashlib.sha256(total).hexdigest(),
                                      "losses": all_losses}, b"" if p["name"] == last else total)
            else:
                for p in self.peers:
                    head, payload = _recv(p["sock"], len(p["shards"]) * n_total * 4)
                    if head.get("step") != step or head.get("shards") != p["shards"]:
                        self._refuse("%s answered the wrong step or shards" % p["name"])
                    if len(payload) != len(p["shards"]) * n_total * 4:
                        self._refuse("%s sent %d gradient bytes" % (p["name"], len(payload)))
                    for i, k in enumerate(p["shards"]):
                        grads[k] = payload[i * n_total * 4:(i + 1) * n_total * 4]
                        losses[k] = head["losses"][i]
                total = ordered_fold(grads[k] for k in range(self.K))
                all_losses = [losses[k] for k in range(self.K)]
                for p in self.peers:
                    _send(p["sock"], {"cmd": "apply", "step": step, "total_sha256": hashlib.sha256(total).hexdigest(),
                                      "losses": all_losses}, total)
            timing["apply_sent"] = round(time.monotonic() - timing["step_sent"], 3)
            hashes = {}
            for p in self.peers:
                head, _ = _recv(p["sock"], 0)
                if head.get("completed") != step + 1:
                    self._refuse("%s did not commit step %d" % (p["name"], step + 1))
                hashes[p["name"]] = head["state"]
                timing["applied_" + p["name"]] = round(time.monotonic() - timing["step_sent"], 3)
            timing["step_seconds"] = round(time.monotonic() - timing.pop("step_sent"), 3)
            if len(set(hashes.values())) != 1:
                self._refuse("replicas disagree after step %d: %s" % (step + 1, hashes))
            row = dict(step=step + 1, losses=[losses[k] for k in range(self.K)],
                       state=next(iter(hashes.values())), workers=hashes,
                       vendors={p["name"]: p.get("vendor") for p in self.peers},
                       total_sha256=hashlib.sha256(total).hexdigest(), protocol="chained" if self.chained else "gathered",
                       timing=timing)
            rows.append(row)
            if self.on_step:
                self.on_step(row)
        for p in self.peers:
            _send(p["sock"], {"cmd": "done"})
        return rows


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
        self.trainer, self.shards, self.batches = trainer, sorted(int(k) for k in shards), batches
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
        n_total = len(memoryview(state["parameters"]).cast("B")) // 4
        sock = self._connect()
        committed = 0
        held, total, on_device = [], None, False  # chained: this block's gradients until its turn, then the fold
        try:
            _send(sock, dict(protocol=PROTOCOL, name=self.name, vendor=vendor(), shards=self.shards,
                             completed=tr.step_, state=state_hash(state), n_total=n_total, chained=self.chained))
            while True:
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
                        for k in self.shards:
                            losses.append(tr.shard_gradient_fold(self.batches(head["step"], k)))
                        held, total, on_device = [], None, True
                    else:
                        for k in self.shards:
                            loss, g = tr.shard_gradient(self.batches(head["step"], k))
                            losses.append(loss)
                            parts.append(memoryview(g).cast("B").tobytes())
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
                        for g in held:
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
                    total = None
                    committed += 1
                    digest = trainer_state_hash(tr)
                    if self.on_commit is not None:
                        self.on_commit(tr.step_, dict(step=tr.step_, state=digest, total_sha256=want,
                                                      losses=head.get("losses")))
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
        print("step %d state %s %s" % (row["step"], row["state"][:16], sorted(row["vendors"].items())), flush=True)
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(line + "\n")

    rows = Coordinator(host=args.host, port=args.port, workers=args.workers, logical_shards=args.shards,
                       steps=args.steps, on_step=on_step, chained=args.chained).run()
    print("AGREED on %d steps across %d workers" % (len(rows), args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
