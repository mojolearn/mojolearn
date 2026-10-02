#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The schedule block table against the per-step exact routes, every step
(lane gap-train-utils, 2026-10-02).

`_training_impl._LrTable.lr_at` answers runs of steps from binary64 values
with rigorous error bounds and falls back to the exact route per step; its
answer must be the exact route's float32 bit for bit. This driver walks the
board's schedules (tools/bench_board_extra.py SCHED) and edge shapes (short
spans, warmup 0, total == warmup, three phases, linear annealing, pct_start
at 0 and 1) over every step, compares `lr_at(t)` with `_lr_at_slow(t)`, and,
on a stride, with the slow route under MOJOLEARN_LR_EXACT_ONLY=1 (the pure
rational route). Prints one line per case with the fallback count and
`SCHED_TABLE_CHECK PASS` or `... FAIL`; exits 1 on FAIL. Host Python only;
runs through `lq add <box> CMD`.
"""
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojolearn import _training_impl as T  # noqa: E402
from mojolearn._x_sequence_sched import OneCycleLR  # noqa: E402


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def cases():
    n = 100_000
    yield "lr-constant", T.ConstantLR(peak_lr=1e-3, warmup_steps=1000), n
    yield "lr-warmup-linear", T.WarmupLinearLR(1e-3, warmup_steps=1000, total_steps=n, min_lr=1e-5), n
    yield "lr-warmup-cosine", T.WarmupCosineLR(1e-3, warmup_steps=1000, total_steps=n, min_lr=1e-5), n
    yield "lr-onecycle", OneCycleLR(max_lr=0.1, total_steps=n, pct_start=0.3, anneal_strategy="cos",
                                    div_factor=25.0, final_div_factor=1e4, three_phase=False), n + 1
    yield "onecycle-linear", OneCycleLR(0.1, 20_000, anneal_strategy="linear"), 20_001
    yield "onecycle-3phase", OneCycleLR(3e-4, 30_001, pct_start=0.25, three_phase=True), 30_002
    yield "onecycle-pct0", OneCycleLR(0.05, 5000, pct_start=0.0), 5001
    yield "onecycle-pct1", OneCycleLR(0.05, 5000, pct_start=1.0), 5001
    for tot in (1, 2, 7, 10, 50, 333):
        yield "onecycle-%d" % tot, OneCycleLR(0.1, tot), tot + 1
        yield "onecycle-lin-%d" % tot, OneCycleLR(0.1, tot, anneal_strategy="linear"), tot + 1
    yield "constant-w0", T.ConstantLR(2e-4), 3000
    yield "linear-w0", T.WarmupLinearLR(3e-3, 0, 5000, 0.0), 6000
    yield "cosine-w0", T.WarmupCosineLR(3e-3, 0, 5000, 0.0), 6000
    yield "linear-eq", T.WarmupLinearLR(1e-3, 300, 300, 1e-6), 1000
    yield "cosine-eq", T.WarmupCosineLR(1e-3, 300, 300, 1e-6), 1000
    for span in (1, 2, 5, 9, 64):
        yield "linear-span%d" % span, T.WarmupLinearLR(7e-4, 10, 10 + span, 3e-6), 10 + span + 300
        yield "cosine-span%d" % span, T.WarmupCosineLR(7e-4, 10, 10 + span, 3e-6), 10 + span + 300
    yield "linear-nototal", T.WarmupLinearLR(1e-3, 77), 3000
    yield "cosine-big", T.WarmupCosineLR(0.37, 4321, 1_000_003, 0.0013), 200_000


def main():
    fails = 0
    for name, sch, last in cases():
        slow_calls = [0]
        slow = sch._lr_at_slow

        def counted(t, slow=slow):
            slow_calls[0] += 1
            return slow(t)
        sch._lr_at_slow = counted
        bad = None
        for t in range(1, last + 1):
            a = sch.lr_at(t)
            if bits(a) != bits(slow(t)):
                bad = (t, a, slow(t))
                break
        ex_bad = None
        if bad is None:
            os.environ["MOJOLEARN_LR_EXACT_ONLY"] = "1"
            try:
                stride = max(1, last // 4000)
                for t in list(range(1, last + 1, stride)) + [last]:
                    if bits(sch.lr_at(t)) != bits(slow(t)):
                        ex_bad = (t, sch.lr_at(t), slow(t))
                        break
            finally:
                os.environ.pop("MOJOLEARN_LR_EXACT_ONLY", None)
        st = "ok" if bad is None and ex_bad is None else "FAIL"
        fails += st != "ok"
        print("SCHED %s steps=%d slow_calls=%d %s%s" % (name, last, slow_calls[0], st,
              "" if st == "ok" else " first=%r exact=%r" % (bad, ex_bad)), flush=True)
    print("SCHED_TABLE_CHECK %s fails=%d" % ("PASS" if not fails else "FAIL", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
