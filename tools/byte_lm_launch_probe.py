#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The per-launch host cost of the byte LM's device context with a resident
session OPEN at the board shape, against a FRESH context (lane/neural-pass43,
2026-10-01). On Metal every live, separately allocated buffer is made
resident on every launch, so a session holding ~1,300 buffers pays per
launch what a fresh context does not; the arena (core/device_arena.mojo)
carves the session's buffers from a few chunks instead. Prints both numbers
and the arena pool's state, so a gain on the LM cells is attributable.

    python tools/byte_lm_launch_probe.py                              # board shape, 2000 launches
    MOJOLEARN_DEVICE_ARENA=0 python tools/byte_lm_launch_probe.py     # the restore
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--shape", default="full", help="bench_board_neural LM shape (default full)")
    args = ap.parse_args(argv)
    import numpy as np
    import bench_board_neural as bbn
    import mojolearn
    binding = mojolearn._byte_lm_impl._load()
    fresh = binding.byte_lm_launch_probe(args.count)
    print("fresh context: %.1f us per enqueue, %.1f us per launch with the wait" % (fresh[0], fresh[1]), flush=True)
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "lm-forward.npz")
        bbn.make_inputs("lm-forward", args.shape, 2, path)
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        runner = bbn.build_runner("lm-forward", "ours", args.shape, data)
        runner.call()
        tr = runner.trainer
        probe = tr._session_binding.byte_lm_session_launch_probe(tr._native_session, args.count)
        dims = bbn.lm_dims("lm-forward", args.shape)
        print("open session (B%d L%d, %d layers): %.1f us per enqueue, %.1f us per launch with the wait; "
              "arena %s, chunks %d (%d in use, %d floats), views %d"
              % (dims[0], dims[1], dims[7], probe[0], probe[1], "on" if probe[2] else "off",
                 probe[3], probe[4], probe[5], probe[6]), flush=True)
        print("LAUNCH_PROBE fresh_us %.2f open_us %.2f arena %d views %d" % (fresh[0], probe[0], probe[2], probe[6]), flush=True)
        if hasattr(tr, "close"):
            tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
