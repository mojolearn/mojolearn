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

__all__ = ["ordered_fold", "fold_pair", "state_hash", "Coordinator", "Worker", "CrossVendorMismatch", "PROTOCOL", "BatchFeed", "LearningRateTable"]


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
    return _native_cross_vendor("trainer_hash", (trainer,), _cross_vendor_errors)


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

    def drain_events(self):
        """Return and clear committed step notifications currently available."""
        return _native_cross_vendor("drain_events", (self,))


class BatchFeed:
    """Own int32 batches arranged as (steps, shards, batch, length + 1).

    ``start_step`` is the absolute training step of the first table row.
    Calling the feed returns an owned batch for one absolute step and shard.
    """
    def __init__(self, batches, *, start_step=0):
        return _native_cross_vendor("feed_init", (self, batches, start_step))

    def __call__(self, step, shard):
        return _native_cross_vendor("feed_batch", (self, step, shard))


class LearningRateTable:
    """Own positive float32 learning rates indexed by absolute training step."""
    def __init__(self, rates, *, start_step=0):
        return _native_cross_vendor("rates_init", (self, rates, start_step))

    def __call__(self, step):
        return _native_cross_vendor("rates_value", (self, step))


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
        return _native_cross_vendor("worker_init", (self, state, shards, batches, address, name, device, trainer, connect_timeout, timeout, chained, lr_for_step, on_commit), _cross_vendor_errors)

    def _connect(self):
        return _native_cross_vendor("worker_connect", (self,), _cross_vendor_errors)

    def run(self):
        """Serve until the coordinator says done; returns the steps committed."""
        return _native_cross_vendor("worker_run", (self,), _cross_vendor_errors)

    def drain_events(self):
        """Return and clear commit metadata, including losses and state hashes."""
        return _native_cross_vendor("drain_events", (self,))


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


# ABI setup connects concrete native-owned classes and the selected vendor.
from .parallel_training import ParallelByteLanguageModelTrainer as _ParallelByte
from ._backend import vendor as _installed_vendor
_native_cross_vendor("worker_initialize", (_ParallelByte, BatchFeed, LearningRateTable, _installed_vendor()))


if __name__ == "__main__":
    sys.exit(main())
