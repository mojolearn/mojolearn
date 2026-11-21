# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the whole-fold process coordination (RUN): the worker
indices and the installed route, the worker's request codes and their
order rule, the wave plan and the fold order of a reply batch.
`bindings/parallel_cv_native_door.mojo` unpacks the Python values into these
arguments, moves the frames and packs the scores; no Python object reaches
this file. The folds run in the workers; what is here is a few words per
wave: nothing to split across cores."""

from core.parallel_worker_request import WorkerRequest

# A CV worker request: the `op` of its typed request
# (core/parallel_worker_request.mojo). A fold request's handle is the fold
# index and its one word is the capture flag; the others carry no word.
# REMOVED: the request names on the wire and code 2, the "fold" request that
# carried a fold's rows (no coordinator sent it since the rows became resident).
comptime CV_REQUEST_IDENTITY = 1
comptime CV_REQUEST_DATA = 3
comptime CV_REQUEST_FOLD_ROWS = 4

#: UNPACK hands a worker index that is not an exact int as this value: it is
#: refused in the words of a negative index, at the position it stands in.
comptime CV_DEVICE_NOT_AN_INDEX = -1


def cv_devices_admit(devices: List[Int]) raises:
    """Distinct nonnegative worker indices, refused in order of position."""
    if len(devices) == 0:
        raise Error("ValueError: devices must not be empty")
    for i in range(len(devices)):  # small-loop(explicit worker indices, before spawn)
        if devices[i] < 0:
            raise Error("ValueError: devices must contain distinct nonnegative integers")
        for j in range(i):  # small-loop(earlier worker indices)
            if devices[j] == devices[i]:
                raise Error("ValueError: devices must not repeat a worker index")


def cv_vendor_admit(devices: List[Int], vendor: String, cpu_only: Bool) raises:
    """The routes whole-fold workers exist for: CUDA, HIP, Metal device
    zero, or a CPU-only install."""
    if vendor == "metal" and (len(devices) != 1 or devices[0] != 0):
        raise Error("NotImplementedError: Metal cross-validation supports devices=(0,) only")
    if vendor != "cuda" and vendor != "hip" and vendor != "metal" and not (vendor == "cpu" and cpu_only):
        raise Error("NotImplementedError: parallel cross-validation requires CUDA, HIP, Metal device zero or a CPU-only install")


def cv_plain_words(op: Int, count: Int) -> List[List[Int64]]:
    """`count` typed requests of one wordless operation (identity, data):
    one per worker."""
    var wire = WorkerRequest(op).encode()
    var words = List[List[Int64]]()
    for _ in range(count):  # small-loop(one request per selected worker)
        words.append(wire.copy())
    return words^


def cv_fold_request(fold: Int, capture: Bool) -> List[Int64]:
    """The typed request of one fold: its index and the capture flag."""
    var request = WorkerRequest(CV_REQUEST_FOLD_ROWS, fold)
    request.words.append(1 if capture else 0)
    return request.encode()


def cv_fold_admit(request: WorkerRequest) raises:
    if request.handle < 0 or len(request.words) != 1 or (request.words[0] != 0 and request.words[0] != 1):
        raise Error("ValueError: malformed native CV fold request")


def cv_request_admit(code: Int, identified: Bool) raises:
    """A worker fits nothing and keeps no rows before its identity was
    admitted; an unknown request is refused."""
    if code != CV_REQUEST_IDENTITY and code != CV_REQUEST_DATA and code != CV_REQUEST_FOLD_ROWS:
        raise Error("ValueError: unknown native CV worker request")
    if code != CV_REQUEST_IDENTITY and not identified:
        raise Error("RuntimeError: CV worker identity must be admitted before fitting")


def cv_wave_count(width: Int, folds: Int, start: Int) -> Int:
    """Folds of the wave that starts at fold `start`: one per worker, fewer
    in the last wave."""
    return min(width, folds - start)


def cv_batch_count_admit(listed: Bool, count: Int, expected: Int) raises:
    if not listed or count != expected:
        raise Error("ValueError: cross-validation workers returned an incomplete fold batch")


def cv_reply_order_admit(shaped: Bool, index: Int, expected: Int) raises:
    """`shaped`: the reply is a 3-tuple whose first field is an exact int
    (`index`). Replies are consumed in fold order, never completion order."""
    if not shaped or index != expected:
        raise Error("ValueError: cross-validation worker fold order differs")


# The stages of one run, in the order a complete run passes them.
comptime CV_STAGE_VALIDATE = 0
comptime CV_STAGE_VALIDATED = 1
comptime CV_STAGE_START = 2
comptime CV_STAGE_IDENTITY = 3
comptime CV_STAGE_DATA = 4
comptime CV_STAGE_WAVE = 5
comptime CV_STAGE_CLOSE = 6

comptime CV_STATUS_VALIDATING = 0
comptime CV_STATUS_RUNNING = 1
comptime CV_STATUS_FAILED = 2
comptime CV_STATUS_COMPLETE = 3


def cv_stage_name(stage: Int) -> String:
    if stage == CV_STAGE_VALIDATE:
        return "validate"
    if stage == CV_STAGE_VALIDATED:
        return "validated"
    if stage == CV_STAGE_START:
        return "start"
    if stage == CV_STAGE_IDENTITY:
        return "identity"
    if stage == CV_STAGE_DATA:
        return "data"
    if stage == CV_STAGE_WAVE:
        return "wave"
    return "close"


def cv_status_name(status: Int) -> String:
    if status == CV_STATUS_VALIDATING:
        return "VALIDATING"
    if status == CV_STATUS_RUNNING:
        return "RUNNING"
    if status == CV_STATUS_FAILED:
        return "FAILED"
    return "COMPLETE"


struct CvTrace(Movable):
    """The record of one whole-fold run, kept as words while the run is in
    progress: the stages passed, the workers' process ids, the size of each
    wave, one (fold, device, process id) triple per assigned fold, the status,
    whether the pool is closed and the failure text. The door publishes it as
    the `last_run` mapping when validation ends and when the run ends."""
    var stages: List[Int]
    var workers: List[Int]
    var waves: List[Int]
    var assignments: List[Int]
    var status: Int
    var closed: Bool
    var error: String

    def __init__(out self):
        self.stages = List[Int]()
        self.stages.append(CV_STAGE_VALIDATE)
        self.workers = List[Int]()
        self.waves = List[Int]()
        self.assignments = List[Int]()
        self.status = CV_STATUS_VALIDATING
        self.closed = True
        self.error = String()

    def stage(mut self, stage: Int):
        self.stages.append(stage)

    def assign(mut self, fold: Int, device: Int, pid: Int):
        self.assignments.append(fold)
        self.assignments.append(device)
        self.assignments.append(pid)

    def finish(mut self, status: Int, closed: Bool):
        self.closed = closed
        self.status = status
        self.stages.append(CV_STAGE_CLOSE)
