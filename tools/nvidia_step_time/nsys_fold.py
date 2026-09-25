#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Fold an `nsys stats -r cuda_gpu_trace -f csv` file of a lean B4 step run
into one line per kernel name (lane/nvidia-step-time, 2026-09-25).

    python3 tools/nvidia_step_time/nsys_fold.py <trace.csv> [--all]

The run traces two lean steps; the first carries setup. The LAST step is cut
at the largest idle gap between consecutive GPU activities in the middle of
the timeline (the host work between two steps), and only activities after it
are folded unless --all. Columns: name (demangled prefix), launches, total
ms, mean us, share of the folded kernel time, registers per thread, static
and dynamic shared bytes, grid and block of the first launch. Memory copies
and sets fold under their own names.
"""
import csv
import re
import sys
import zlib
from collections import defaultdict


def short(name):
    n = name.strip()
    # Mojo kernel symbols carry the module path and the parameter list; keep
    # the function name and a hash of the rest so instantiations stay apart.
    m = re.search(r"(identical_gemm_[a-z0-9_]+|fused_[a-z0-9_]+|[a-z_]*kernel[a-z0-9_]*)", n)
    base = m.group(1) if m else n[:60]
    return base[:70] + "#" + format(zlib.crc32(n.encode()) & 0xFFFF, "04x") if base != n else base


def col(header, *prefixes):
    for p in prefixes:
        for i, h in enumerate(header):
            if h.startswith(p):
                return i
    return None


def main(argv):
    path = argv[1]
    use_all = "--all" in argv
    rows = list(csv.reader(open(path, newline="", errors="replace")))
    header = rows[0]
    i_start, i_dur = col(header, "Start"), col(header, "Duration")
    i_name = col(header, "Name")
    i_reg = col(header, "Reg/Trd")
    i_ss, i_ds = col(header, "StcSMem"), col(header, "DymSMem")
    i_g = [col(header, "GrdX"), col(header, "GrdY"), col(header, "GrdZ")]
    i_b = [col(header, "BlkX"), col(header, "BlkY"), col(header, "BlkZ")]
    ev = []
    for r in rows[1:]:
        try:
            ev.append((int(float(r[i_start])), int(float(r[i_dur])), r))
        except (ValueError, IndexError, TypeError):
            continue
    ev.sort(key=lambda e: e[0])
    cut = 0
    if not use_all and len(ev) > 10:
        lo, hi = len(ev) // 5, 4 * len(ev) // 5
        best, gap = lo, -1
        for j in range(lo, hi):
            g = ev[j][0] - (ev[j - 1][0] + ev[j - 1][1])
            if g > gap:
                best, gap = j, g
        cut = best
    sel = ev[cut:]
    agg = defaultdict(lambda: [0, 0, None])
    for s, d, r in sel:
        k = short(r[i_name])
        a = agg[k]
        a[0] += 1
        a[1] += d
        if a[2] is None:
            def g(i):
                return r[i] if i is not None and i < len(r) else ""
            a[2] = (g(i_reg), g(i_ss), g(i_ds), "x".join(g(i) for i in i_g), "x".join(g(i) for i in i_b), r[i_name][:300])
    total = sum(a[1] for a in agg.values())
    span = (sel[-1][0] + sel[-1][1] - sel[0][0]) if sel else 0
    print("# folded %d of %d activities (cut at %d); kernel+copy time %.3f ms; wall span %.3f ms" % (len(sel), len(ev), cut, total / 1e6, span / 1e6))
    print("name\tlaunches\ttotal_ms\tmean_us\tshare\treg_per_thread\tstatic_smem\tdyn_smem\tgrid\tblock\tfull_name")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        reg, ss, ds, grid, blk, full = a[2]
        print("%s\t%d\t%.3f\t%.1f\t%.4f\t%s\t%s\t%s\t%s\t%s\t%s" % (k, a[0], a[1] / 1e6, a[1] / a[0] / 1e3, a[1] / total if total else 0, reg, ss, ds, grid, blk, full))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
