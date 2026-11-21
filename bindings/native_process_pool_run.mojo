# SPDX-License-Identifier: Apache-2.0
"""The typed state of the native process pool (RUN): the pool record, the
rules of a request wave and the per-worker reap outcome.

No Python object is read here. The door (`bindings/native_process_pool_door.mojo`)
holds one `ProcessPoolRecord` per pool in a native owner; the pool dictionary a
caller sees is a mirror the door publishes after a start, a wave and a close.

A wave is serial by nature on the coordinator's side: one request frame is
written to each worker's socket, then one reply is read from each, in request
order. The workers are separate processes and compute at the same time; the
coordinator only waits on their sockets, so there is no work here to split
across cores.
"""
from core.parallel_worker_frame import WIRE_RESULT, WIRE_ERROR


struct ProcessPoolRecord(Movable):
    """One pool: the selected physical devices, whether one worker owns all of
    them (`cooperative`) or each device has its own, the closed flag, the next
    request sequence number and, per started worker in spawn order, its
    process id and the descriptor of its authenticated socket (-1 until its
    hello is admitted, and again after its socket is closed)."""
    var cooperative: Bool
    var closed: Bool
    var sequence: Int
    var devices: List[Int]
    var pids: List[Int]
    var fds: List[Int]

    def __init__(out self, var devices: List[Int], cooperative: Bool):
        self.cooperative = cooperative
        self.closed = False
        self.sequence = 0
        self.devices = devices^
        self.pids = List[Int]()
        self.fds = List[Int]()

    def width(self) -> Int:
        """The number of workers a started pool has."""
        return 1 if self.cooperative else len(self.devices)

    def started(self) -> Bool:
        return len(self.pids) != 0


def pool_index_admit(width: Int, worker_index: Int) raises:
    if worker_index < 0 or worker_index >= width:
        raise Error("process: addressed worker index outside the selected group")


def pool_wave_width(workers: Int, remaining: Int) -> Int:
    """Requests of the next wave: one per worker until the requests run out."""
    return workers if workers < remaining else remaining


def pool_reply_failed(kind: Int, sequence: Int, expected: Int, addressed: Bool) raises -> Bool:
    """Admits one reply frame against the request it answers. Returns whether
    the worker reported a model error (its record is then the error text)."""
    if sequence != expected:
        raise Error("process: response sequence mismatch")
    if kind != WIRE_RESULT and kind != WIRE_ERROR:
        if addressed:
            raise Error("process: unexpected addressed worker response")
        raise Error("process: unexpected worker response")
    return kind == WIRE_ERROR


struct ReapOutcome(Movable):
    """Per worker, in pool order: the reap's error ("" when reaped), the
    seconds from the start of the reap to it, and the signal it needed."""
    var errors: List[String]
    var seconds: List[Float64]
    var signal: List[Int]

    def __init__(out self, n: Int):
        self.errors = List[String](length=n, fill=String())
        self.seconds = List[Float64](length=n, fill=0.0)
        self.signal = List[Int](length=n, fill=0)
