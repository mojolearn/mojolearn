#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU busy fraction and dispatch profile for one extratrees fit.

    python3 extratrees/bench/profile_et_metal.py /tmp/et_profile.trace

Reads a Metal System Trace recorded over `build/et_fit_once` and answers the
one question DEVIATIONS 212/213 left open: at the shipped size, is the
device BUSY (the kernels are the floor) or IDLE between dispatches (the cost
is host-side submission)?

The XML parser is the RF lane's technique (`ensemble/bench/profile_metal.py`
-- that file is another lane's working copy, so the technique is reused in
this lane's own file rather than by editing theirs): Instruments interns its
strings (first occurrence carries id=, later rows carry ref= with no text),
and label columns keep the real value in the `fmt` attribute with empty
text. Per-kernel attribution is NOT available from the stock template (the
encoders are unnamed); busy fraction and dispatch-size distribution are.
"""

import collections
import subprocess
import sys
import xml.etree.ElementTree as ET


def pm_table(trace, schema):
    xml = subprocess.run(
        ["xctrace", "export", "--input", trace, "--xpath",
         '/trace-toc/run[@number="1"]/data/table[@schema="%s"]' % schema],
        capture_output=True, text=True,
    ).stdout
    if "<row>" not in xml:
        return [], []
    root = ET.fromstring(xml)
    node = root.find("node")
    cols = [c.find("mnemonic").text for c in node.find("schema").findall("col")]
    pool, rows = {}, []
    for row in node.findall("row"):
        rec = {}
        for i, child in enumerate(row):
            key = cols[i] if i < len(cols) else child.tag
            ref = child.attrib.get("ref")
            if ref is not None:
                val = pool.get(ref, (None, None))
            else:
                val = (child.text, child.attrib.get("fmt"))
                if "id" in child.attrib:
                    pool[child.attrib["id"]] = val
            rec[key] = val[0] if val[0] is not None else val[1]
        rows.append(rec)
    return cols, rows


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "/tmp/et_profile.trace"
    who = sys.argv[2] if len(sys.argv) > 2 else "et_fit_once"

    _c, gi = pm_table(trace, "metal-gpu-intervals")
    if not gi:
        print("no metal-gpu-intervals rows -- was the trace recorded?")
        return 1
    ours = [r for r in gi if who in (r.get("event-label") or "")]
    comp = [r for r in ours if r.get("channel-name") == "Compute"]
    if not comp:
        # Some traces label the channel differently; fall back to all rows
        # for the process and say so.
        comp = ours
        print("NOTE: no channel-name == Compute rows; using all %d process"
              " rows" % len(ours))
    dur = [int(r["duration"]) for r in comp]
    beg = [int(r["start"]) for r in comp]
    busy = sum(dur)
    span = max(b + d for b, d in zip(beg, dur)) - min(beg)

    print("compute dispatches      %d" % len(dur))
    print("GPU busy (sum)          %.1f ms" % (busy / 1e6))
    print("span, first to last     %.1f ms" % (span / 1e6))
    print("GPU BUSY FRACTION       %.1f%%" % (100.0 * busy / span if span else 0))
    q = sorted(dur)
    sd = sorted(dur, reverse=True)
    print("dispatch mean %.1f us  median %.1f us  p90 %.1f us  max %.1f us"
          % (sum(dur) / len(dur) / 1e3, q[len(q) // 2] / 1e3,
             q[int(0.9 * len(q))] / 1e3, max(dur) / 1e3))
    for k in (10, 50, 200, 1000, 5000):
        if k < len(sd):
            print("the %5d longest dispatches hold %5.1f%% of busy time"
                  % (k, 100.0 * sum(sd[:k]) / busy))

    _c, cb = pm_table(trace, "metal-application-command-buffer-submissions")
    ourcb = [r for r in cb if who in (r.get("process") or "")]
    if ourcb:
        ne = [int(r["num-encoders"]) for r in ourcb if r.get("num-encoders")]
        print("command buffer submissions  %d" % len(ourcb))
        if ne:
            print("encoders per buffer         mean %.2f, max %d"
                  % (sum(ne) / len(ne), max(ne)))

    for schema, title in (("device-thermal-state-intervals", "thermal"),
                          ("gpu-performance-state-intervals", "gpu perf")):
        _c, trows = pm_table(trace, schema)
        if trows:
            seen = collections.Counter()
            for r in trows:
                vals = [v for k, v in r.items()
                        if k not in ("start", "duration") and v and v != "None"]
                seen[" ".join(vals[:2])] += int(r.get("duration") or 0)
            print("%s states:" % title)
            for state, ns in seen.most_common(4):
                print("   %-40s %8.1f ms" % (state[:40], ns / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
