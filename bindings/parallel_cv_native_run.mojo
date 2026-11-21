# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the whole-fold process coordination (RUN): the worker
indices and the installed route, the worker's request codes and their
order rule, the wave plan and the fold order of a reply batch.
`bindings/parallel_cv_native_door.mojo` unpacks the Python values into these
arguments, moves the frames and packs the scores; no Python object reaches
this file. The folds run in the workers; what is here is a few words per
wave: nothing to split across cores."""

# A CV worker request, by its name on the wire.
comptime CV_REQUEST_UNKNOWN = 0
comptime CV_REQUEST_IDENTITY = 1
comptime CV_REQUEST_FOLD = 2
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


def cv_request_code(name: String, length: Int) -> Int:
    """The request a worker received, from its name and its field count
    (an identity request has no field)."""
    if name == "identity" and length == 1:
        return CV_REQUEST_IDENTITY
    if name == "fold":
        return CV_REQUEST_FOLD
    if name == "data":
        return CV_REQUEST_DATA
    if name == "fold_rows":
        return CV_REQUEST_FOLD_ROWS
    return CV_REQUEST_UNKNOWN


def cv_request_admit(code: Int, identified: Bool) raises:
    """A worker fits nothing and keeps no rows before its identity was
    admitted; an unknown request is refused."""
    if code == CV_REQUEST_UNKNOWN:
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
