#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where a step's `hash_seconds` goes on the host, at the GPT-3 Small shape
(162,147,840 parameters: parameters, m, v and the summed gradient at 648.6 MB
each, flags one int32 per tensor), with no device: the parts a Mac can run.

    python3 host_split.py [--n 162147840] [--tensors 147] [--hold 3]

  alloc       `empty()` of the four 648.6 MB arrays, what `export_raw` and
              `export_gradients` do every step (zero-filled, fresh pages)
  host_copy   one memmove of 2.59 GB into already-touched memory, a floor for
              the binding's final copy into the Python buffers (the device
              copy and the binding's element loop are not here)
  old_hash    `_hash_arrays(v2)` + `_hash_gradient(v2)`, the runner before
              2026-09-25: eight slice threads, one array at a time
  new_hash    `Digests(v2)`: every slice of every array and the gradient at once
  one_core    one sha256 over the same 2.59 GB (the CPU cost of the digest)
  overlap     `Digests` started, then the interpreter lock HELD in one C call
              for --hold seconds (the native step keeps it): how long after
              the call returns the digests are ready

Run with the package importable (a checkout: PYTHONPATH=python, or the
wheel) from the repository root; the digests are also checked equal.
"""
import argparse
import ctypes
import hashlib
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))


def _package():
    try:
        from mojolearn._buffer import empty  # noqa: F401
    except ImportError:  # a checkout without built binaries: the pure-Python modules alone
        pkg = types.ModuleType("mojolearn")
        pkg.__path__ = [str(ROOT / "python" / "mojolearn")]
        sys.modules["mojolearn"] = pkg
    from mojolearn._buffer import empty
    return empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=162147840)
    ap.add_argument("--tensors", type=int, default=147)
    ap.add_argument("--hold", type=float, default=3.0)
    args = ap.parse_args()
    empty = _package()
    import lm_segment as seg
    out = {}
    t = time.perf_counter()
    raw = {k: empty((args.n,), "<f4") for k in ("parameters", "m", "v")}
    raw["flags"] = empty((args.tensors,), "<i4")
    grad = empty((args.n,), "<f4")
    out["alloc"] = time.perf_counter() - t
    # distinct contents, so no digest is of zeros
    for i, a in enumerate((raw["parameters"], raw["m"], raw["v"], grad)):
        mv = seg._bytes_of(a, writable=True)
        mv[:4096] = bytes([i + 1]) * 4096
        mv[-4096:] = bytes([i + 7]) * 4096
    from mojolearn._buffer import addr, addr_ro
    t = time.perf_counter()
    for a in (raw["parameters"], raw["m"], raw["v"]):  # 3 x 648.6 MB into the touched gradient buffer ...
        ctypes.memmove(addr(grad, name="dst"), addr_ro(a, name="src"), a.nbytes)
    ctypes.memmove(addr(raw["v"], name="dst"), addr_ro(raw["m"], name="src"), grad.nbytes)  # ... and one more
    out["host_copy"] = time.perf_counter() - t
    mv = seg._bytes_of(grad, writable=True)
    mv[:4096] = bytes([9]) * 4096
    t = time.perf_counter()
    old = (seg._hash_arrays(raw, seg.SCHEME_V2), seg._hash_gradient(grad, seg.SCHEME_V2))
    out["old_hash"] = time.perf_counter() - t
    t = time.perf_counter()
    new = seg.Digests(raw, grad, seg.SCHEME_V2).result()
    out["new_hash"] = time.perf_counter() - t
    assert new == old, (new, old)
    t = time.perf_counter()
    h = hashlib.sha256()
    for a in (raw["parameters"], raw["m"], raw["v"], grad):
        h.update(seg._bytes_of(a))
    out["one_core"] = time.perf_counter() - t
    usleep = ctypes.PyDLL(None).usleep  # a PyDLL call keeps the interpreter lock
    usleep.argtypes = [ctypes.c_uint]
    t = time.perf_counter()
    d = seg.Digests(raw, grad, seg.SCHEME_V2)
    started = time.perf_counter() - t
    usleep(int(args.hold * 1e6))
    back = time.perf_counter()
    again = d.result()
    out["overlap_start"] = started
    out["overlap_after_hold"] = time.perf_counter() - back
    out["overlap_all_started"] = d.started_all
    assert again == old
    print("n=%d (%.1f MB an array), digests equal: %s" % (args.n, 4 * args.n / 1e6, old[0][:16]))
    for k, v in out.items():
        print("%-20s %s" % (k, ("%.3f s" % v) if isinstance(v, float) else v))



if __name__ == "__main__":
    main()
