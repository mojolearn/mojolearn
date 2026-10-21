# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native worker/coordinator connections for replicated training.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
"""
import sys

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
    """Return SHA256 over a native trainer's current complete replica state."""
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
    the agreed state hash and each worker's hash. ``log_path`` appends each
    agreed step as JSON. Refuses, naming what differs, on a bad split, a
    different starting state or a disagreement after any step."""

    def __init__(self, *, port, workers, logical_shards, steps, host="0.0.0.0",
                 on_step=None, timeout=3600.0, accept_timeout=3600.0, max_gradient_bytes=1 << 31,
                 chained=False, log_path=None):
        return _native_cross_vendor("coordinator_init", (self, port, workers, logical_shards, steps, host, on_step, timeout, accept_timeout, max_gradient_bytes, chained, _NativeReady, log_path), _cross_vendor_errors)

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
    ``batches`` is a :class:`BatchFeed` containing the same logical batches
    on every worker. ``lr_for_step`` is a :class:`LearningRateTable` or None.
    Committed metadata is available through :meth:`drain_events`."""

    def __init__(self, state=None, *, shards, batches, address, name=None, device=0,
                 trainer=None, connect_timeout=600.0, timeout=3600.0, chained=False, lr_for_step=None,
                 on_commit=None, log_path=None, checkpoint_dir=None, checkpoint_steps=(),
                 expected_commits=(), retain_gradients=False, hash_scheme="sha256.v1"):
        return _native_cross_vendor("worker_init", (self, state, shards, batches, address, name, device, trainer, connect_timeout, timeout, chained, lr_for_step, on_commit, log_path, checkpoint_dir, checkpoint_steps, expected_commits, retain_gradients, hash_scheme), _cross_vendor_errors)

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
    return _native_cross_vendor("main", (argv,), _cross_vendor_errors)


# ABI setup connects concrete native-owned classes and the selected vendor.
from .parallel_training import ParallelByteLanguageModelTrainer as _ParallelByte
from ._backend import vendor as _installed_vendor
_native_cross_vendor("worker_initialize", (_ParallelByte, BatchFeed, LearningRateTable, _installed_vendor(), Worker, Coordinator, _NativeReady))


if __name__ == "__main__":
    sys.exit(main())
