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

WHAT IT IS NOT. Not a fast path: every step moves each shard's full
gradient (4 bytes per parameter) to the coordinator and the total back, so it
suits small models on a local network. No authentication or encryption: run
it on a trusted network or through an ssh tunnel. Each worker drives one
device with a replicated optimizer (`ParallelByteLanguageModelTrainer(state,
devices=(d,), pool_optimizer=False)`).
"""
import argparse
import array
import hashlib
import json
import math
import socket
import struct
import sys
import time

PROTOCOL = "mojolearn.cross-vendor.v1"
_TINY = 2.0 ** -126  # smallest normal float32

__all__ = ["ordered_fold", "state_hash", "Coordinator", "Worker", "CrossVendorMismatch", "PROTOCOL"]


class CrossVendorMismatch(RuntimeError):
    """Two workers disagree, or the group is not a valid split of one step."""


# ------------------------------------------------------------ the fold
def _f32(payload):
    return memoryview(payload).cast("B").cast("f")


def ordered_fold(gradients):
    """The device reduction, on the host: copy g[0], then
    total = ftz(ftz(total) + ftz(g[k])) with one float32 rounding per add
    (training/byte_lm_parallel.mojo `_ordered_add_kernel`, fma(1, a, b)).
    `gradients` is a sequence of float32 buffers in shard order; returns
    bytes. Uses numpy when installed; the pure-Python path is exact too,
    because a float64 sum rounded once to float32 is the float32 sum
    (53 >= 2*24 + 2)."""
    gradients = list(gradients)
    if not gradients:
        raise ValueError("ordered_fold needs at least one gradient")
    n = len(_f32(gradients[0]))
    if any(len(_f32(g)) != n for g in gradients):
        raise ValueError("ordered_fold: gradients differ in length")
    try:
        import numpy as np
    except ImportError:
        return _fold_python(gradients, n)
    exp, man, sign = np.uint32(0x7F800000), np.uint32(0x007FFFFF), np.uint32(0x80000000)

    def ftz(x):
        b = x.view(np.uint32)
        return np.where(((b & exp) == 0) & ((b & man) != 0), b & sign, b).view(np.float32)

    total = np.frombuffer(bytes(_f32(gradients[0])), dtype=np.float32).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for g in gradients[1:]:
            total = ftz(ftz(total) + ftz(np.frombuffer(_f32(g), dtype=np.float32)))
    return total.tobytes()


def _ftz(x):
    return math.copysign(0.0, x) if x != 0.0 and abs(x) < _TINY else x


def _fold_python(gradients, n):
    total = array.array("f", _f32(gradients[0]))
    for g in gradients[1:]:
        gv = _f32(g)
        for i in range(n):
            total[i] = _ftz(total[i]) + _ftz(gv[i])
            total[i] = _ftz(total[i])
    return total.tobytes()


def state_hash(state):
    """sha256 over parameters, m, v and flags, in that order."""
    h = hashlib.sha256()
    for key in ("parameters", "m", "v", "flags"):
        h.update(memoryview(state[key]).cast("B"))
    return h.hexdigest()


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
                 on_step=None, timeout=3600.0, accept_timeout=3600.0, max_gradient_bytes=1 << 31):
        if workers < 1 or logical_shards < workers:
            raise ValueError("need 1 <= workers <= logical_shards")
        self.host, self.port, self.n_workers = host, port, workers
        self.K, self.steps, self.on_step = logical_shards, steps, on_step
        self.timeout, self.accept_timeout, self.max_bytes = timeout, accept_timeout, max_gradient_bytes
        self.peers = []

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
        rows = []
        for step in range(int(self.peers[0]["completed"]), self.steps):
            for p in self.peers:
                _send(p["sock"], {"cmd": "step", "step": step})
            grads, losses = {}, {}
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
            for p in self.peers:
                _send(p["sock"], {"cmd": "apply", "step": step}, total)
            hashes = {}
            for p in self.peers:
                head, _ = _recv(p["sock"], 0)
                if head.get("completed") != step + 1:
                    self._refuse("%s did not commit step %d" % (p["name"], step + 1))
                hashes[p["name"]] = head["state"]
            if len(set(hashes.values())) != 1:
                self._refuse("replicas disagree after step %d: %s" % (step + 1, hashes))
            row = dict(step=step + 1, losses=[losses[k] for k in range(self.K)],
                       state=next(iter(hashes.values())), workers=hashes,
                       vendors={p["name"]: p.get("vendor") for p in self.peers})
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
                 trainer=None, connect_timeout=600.0, timeout=3600.0):
        if trainer is None:
            from .parallel_training import ParallelByteLanguageModelTrainer
            trainer = ParallelByteLanguageModelTrainer(state, devices=(device,), logical_shards=1,
                                                       pool_optimizer=False)
        self.trainer, self.shards, self.batches = trainer, sorted(int(k) for k in shards), batches
        self.address, self.connect_timeout, self.timeout = tuple(address), connect_timeout, timeout
        self.name = name or socket.gethostname()

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
        state = tr.state_dict()
        n_total = len(memoryview(state["parameters"]).cast("B")) // 4
        sock = self._connect()
        committed = 0
        try:
            _send(sock, dict(protocol=PROTOCOL, name=self.name, vendor=vendor(), shards=self.shards,
                             completed=tr.step_, state=state_hash(state), n_total=n_total))
            while True:
                head, payload = _recv(sock, n_total * 4)
                cmd = head.get("cmd")
                if cmd == "step":
                    if head["step"] != tr.step_:
                        raise CrossVendorMismatch("%s is at step %d, asked for %d" % (self.name, tr.step_, head["step"]))
                    losses, parts = [], []
                    for k in self.shards:
                        loss, g = tr.shard_gradient(self.batches(head["step"], k))
                        losses.append(loss)
                        parts.append(memoryview(g).cast("B").tobytes())
                    _send(sock, {"step": head["step"], "shards": self.shards, "losses": losses}, b"".join(parts))
                elif cmd == "apply":
                    if len(payload) != n_total * 4:
                        raise CrossVendorMismatch("summed gradient has the wrong size")
                    tr.apply_gradient(array.array("f", payload))
                    committed += 1
                    _send(sock, {"completed": tr.step_, "state": state_hash(tr.state_dict())})
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
    args = ap.parse_args(argv)

    def on_step(row):
        line = json.dumps(row, sort_keys=True)
        print("step %d state %s %s" % (row["step"], row["state"][:16], sorted(row["vendors"].items())), flush=True)
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(line + "\n")

    rows = Coordinator(host=args.host, port=args.port, workers=args.workers, logical_shards=args.shards,
                       steps=args.steps, on_step=on_step).run()
    print("AGREED on %d steps across %d workers" % (len(rows), args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
