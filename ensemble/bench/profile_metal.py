#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS cuML's random forest. Per-file provenance is in this file's own docstring and in NOTICE.
"""Per-kernel GPU time for the random forest, from Apple Instruments.

    ensemble/bench/profile_metal.py record          # trace one fit
    ensemble/bench/profile_metal.py report          # summarize the trace
    ensemble/bench/profile_metal.py record report   # both

WHY INSTRUMENTS AND NOT OUR OWN TIMERS.

Because our own timers have been wrong here before. This repository's
standing note on the subject is blunt: the code was wrong about itself four
times in one day and the instruments failed three times, so device time
comes from Instruments. Wrapping each launch in `synchronize()` and a host
clock would also work, but it SERIALIZES the pipeline it is measuring --
the number it reports is the duration of a kernel that is no longer allowed
to overlap anything, which is a different program. Metal System Trace reads
the GPU's own timeline and perturbs nothing.

WHAT IT REPORTS AND WHAT IT CANNOT.

`metal-gpu-intervals` is device-side: real start times and durations for
work the GPU actually ran, labelled per encoder. Summed per label, that is
the per-kernel profile.

WHAT THIS IS FOR. Before optimizing a kernel, know which one. cuML's
kernels are deliberately plain -- across the whole batched-levelalgo
directory there is not one `__ldg`, `__restrict__` or `__launch_bounds__`,
and exactly one `#pragma unroll` -- so any change we make past them stops
being a port of their algorithm and starts being our own design, which is a
deviation and has to be priced as one. That trade is only worth making
where the time actually is, and until this file existed nobody here knew.

TWO NUMBERS TO READ FIRST, BOTH OF WHICH ARE ABOUT THE CONTROL PLANE
RATHER THAN THE KERNELS:

  * GPU BUSY FRACTION -- summed interval time over wall time. A low
    fraction means the device is idling between launches and the cost is
    host-side dispatch, not arithmetic. A sibling lane on this repo
    measured its GPU round at 279 launches and 38 synchronizations, and
    that, not the kernels, was the bottleneck.
  * INTERVAL COUNT -- how many times each kernel was dispatched. A kernel
    with a small mean and a huge count is a launch-overhead problem and no
    amount of work on its body will help.

The thermal and GPU performance-state tables are printed too, because a
trace taken while the device was throttling describes the throttle.
"""

import argparse
import collections
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE = "/tmp/rf_profile.trace"
BINARY = os.path.join(REPO, "build", "rf_bench")


def record(trace, binary, args, launch_log=None):
    if os.path.exists(trace):
        subprocess.run(["rm", "-rf", trace], check=True)
    cmd = ["xctrace", "record", "--template", "Metal System Trace",
           "--output", trace, "--target-stdout", "-"]
    if launch_log:
        # The workload names its own dispatches: every enqueue site in the
        # forest path calls `core.launch_log.log_launch` first, and with
        # RF_LAUNCH_LOG set the names land in this file in enqueue order.
        # `attr` below joins them to the trace. Truncate first -- a stale
        # log misaligns the join, which then refuses.
        if os.path.exists(launch_log):
            os.remove(launch_log)
        cmd += ["--env", "RF_LAUNCH_LOG=" + launch_log]
    cmd += ["--launch", "--", binary] + args
    print("==> " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        return r.returncode
    # xctrace has been observed writing to the basename in the CWD rather
    # than to --output. Take whichever exists.
    if not os.path.exists(trace) and os.path.exists(os.path.basename(trace)):
        os.replace(os.path.basename(trace), trace)
    return 0


def pm_table(trace, schema):
    """One exported table as (columns, rows-of-dicts).

    TWO THINGS ABOUT INSTRUMENTS XML, BOTH OF WHICH BIT.

    IT INTERNS ITS STRINGS. The first appearance of a value carries
    `id="N"`; every later appearance is `<tag ref="N"/>` with no content.
    Read `.text` alone and you get row one right and every later row empty
    -- which looks like sparse data rather than the parse bug it is.

    AND THE VALUE IS OFTEN IN `fmt`, NOT IN THE TEXT. Numeric columns put
    the number in the text; the label columns put an EMPTY text and the
    real string in the `fmt` attribute. A pool that stores `.text` therefore
    interns `None` for exactly the columns worth reading, and the first
    version of this reported every process as "?" while parsing 9445 rows
    perfectly happily.

    So both are kept and the caller picks.
    """
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
            # Prefer the text (numbers live there); fall back to fmt (which
            # is where every string label lives).
            rec[key] = val[0] if val[0] is not None else val[1]
        rows.append(rec)
    return cols, rows


def report(trace, who="rf_bench"):
    """The profile. Three questions, in the order they should be asked."""
    _c, gi = pm_table(trace, "metal-gpu-intervals")
    if not gi:
        print("no metal-gpu-intervals rows -- was the trace recorded?", file=sys.stderr)
        return 1

    # FILTER ON THE LABEL, NOT ON THE `process` COLUMN. That column is a
    # ref into an id pool that lives in OTHER tables of the same trace, so
    # exporting this table alone leaves every process unresolvable and the
    # first version of this reported 9445 rows belonging to "?". The label
    # carries `( rf_bench (34880) )` inline and needs no pool.
    ours = [r for r in gi if who in (r.get("event-label") or "")]
    comp = [r for r in ours if r.get("channel-name") == "Compute"]
    if not comp:
        print("no Compute rows for %r" % who, file=sys.stderr)
        return 1

    dur = [int(r["duration"]) for r in comp]
    beg = [int(r["start"]) for r in comp]
    busy = sum(dur)
    span = max(b + d for b, d in zip(beg, dur)) - min(beg)

    print("=" * 78)
    print("1. IS THIS A KERNEL PROBLEM AT ALL?")
    print("=" * 78)
    print("  compute dispatches      %d" % len(dur))
    print("  GPU busy (sum)          %.1f ms" % (busy / 1e6))
    print("  span, first to last     %.1f ms" % (span / 1e6))
    print("  GPU BUSY FRACTION       %.1f%%" % (100.0 * busy / span if span else 0))
    print()
    print("  A low busy fraction means the device is IDLE between dispatches and")
    print("  the cost is host-side submission, not arithmetic. Optimizing a kernel")
    print("  body cannot recover time the GPU spent waiting for work.")

    _c, cb = pm_table(trace, "metal-application-command-buffer-submissions")
    ourcb = [r for r in cb if who in (r.get("process") or "")]
    if ourcb:
        ne = [int(r["num-encoders"]) for r in ourcb if r.get("num-encoders")]
        print()
        print("  command buffer submissions  %d" % len(ourcb))
        print("  encoders per buffer         mean %.2f, max %d"
              % (sum(ne) / len(ne), max(ne)))
        print("  submissions with NO encoder %d" % sum(1 for x in ne if x == 0))
        print()
        print("  On Metal every commit is a trip through the driver. One encoder")
        print("  per buffer means nothing is being batched, and CUDA's near-free")
        print("  stream launch -- which is what cuML's design assumes -- is not")
        print("  what this port actually gets.")

    print()
    print("=" * 78)
    print("2. WHERE IS THE GPU TIME, BY DISPATCH SIZE?")
    print("=" * 78)
    sd = sorted(dur, reverse=True)
    q = sorted(dur)
    print("  mean %.1f us   median %.1f us   p90 %.1f us   max %.1f us"
          % (sum(dur) / len(dur) / 1e3, q[len(q) // 2] / 1e3,
             q[int(0.9 * len(q))] / 1e3, max(dur) / 1e3))
    for k in (10, 50, 200, 1000):
        if k < len(sd):
            print("  the %5d longest dispatches hold %5.1f%% of GPU busy time"
                  % (k, 100.0 * sum(sd[:k]) / busy))
    print()
    print("  A median far below the mean means a long tail of tiny dispatches.")
    print("  Those cost a full submission each and return almost no work.")

    print()
    print("=" * 78)
    print("3. WHICH KERNELS ARE LOADED?")
    print("=" * 78)
    _c, sl = pm_table(trace, "metal-shader-profiler-shader-list")
    mine = [r for r in sl if who in (r.get("process") or "")]
    print("  %d distinct compute pipelines" % len(mine))
    for r in mine:
        print("     %s" % (r.get("name") or "?"))
    print()
    print("  PER-KERNEL TIME IS NOT AVAILABLE FROM THIS TRACE. Attribution needs")
    print("  the Shader Timeline instrument, and the stock 'Metal System Trace'")
    print("  template records it Disabled -- the trace's own settings block says")
    print("  so. `metal-gpu-intervals` labels each dispatch by ENCODER, and this")
    print("  port issues one unnamed encoder per launch, so the durations above")
    print("  cannot be joined to these names. Saying which kernel is hot needs")
    print("  either a custom template with Shader Timeline on, or debug labels")
    print("  set on the Metal objects. Until then the numbers in section 1 are")
    print("  the finding, and they point away from the kernels.")

    for schema, title in (("device-thermal-state-intervals", "THERMAL STATE"),
                          ("gpu-performance-state-intervals", "GPU PERF STATE")):
        _c, trows = pm_table(trace, schema)
        if trows:
            seen = collections.Counter()
            for r in trows:
                vals = [v for k, v in r.items()
                        if k not in ("start", "duration") and v and v != "None"]
                seen[" ".join(vals[:2])] += int(r.get("duration") or 0)
            print("\n%s during the trace:" % title)
            for state, ns in seen.most_common(6):
                print("   %-44s %8.1f ms" % (state[:44], ns / 1e6))
    return 0


def attr(trace, logpath, who="rf_bench"):
    """PER-SITE GPU TIME, from the launch log joined to the trace.

    Apple's stock template records the Shader Timeline instrument
    Disabled and `xctrace` cannot enable it (`--instrument 'GPU'` records
    nothing into `gpu-shader-profiler-interval` either -- measured
    2026-08-22), so the trace knows every dispatch's DURATION but not its
    NAME. The workload's launch log knows every NAME but no duration.
    Metal's compute channel here is a single in-order queue, so device
    order IS enqueue order and line `i` of the log names merged op `i` of
    the trace.

    TWO MEASURED FACTS THE JOIN DEPENDS ON (dispatch_census_probe, prime
    counts 7/11/13/5/17 decomposing uniquely):
      * `enqueue_memset` emits NO Compute interval on this stack (a host
        write on unified memory), so memsets are NOT logged;
      * every `enqueue_copy` direction (H2D, D2D, D2H) IS a Compute
        dispatch, so every copy site IS logged.
    And MAX sometimes expands one enqueue into several dispatches inside
    ONE command buffer (28 of 9405 rows at the 500k shape), so
    consecutive rows sharing a cmdbuffer-id merge into one logical op,
    durations summed. After the merge the counts must match EXACTLY or
    the join is REFUSED: one slip mislabels every row after it.
    """
    names = [l.strip() for l in open(logpath) if l.strip()]
    _c, gi = pm_table(trace, "metal-gpu-intervals")
    comp = [r for r in gi
            if who in (r.get("event-label") or "")
            and r.get("channel-name") == "Compute"]
    comp.sort(key=lambda r: int(r["start"]))
    merged = []
    prev_cb = object()
    for r in comp:
        cb = r.get("cmdbuffer-id")
        if cb == prev_cb and merged:
            merged[-1] += int(r["duration"])
        else:
            merged.append(int(r["duration"]))
        prev_cb = cb
    print("logged %d   trace Compute rows %d   merged ops %d"
          % (len(names), len(comp), len(merged)))
    if len(merged) != len(names):
        print("COUNT MISMATCH after cmdbuf merge -- join refused. Was the",
              file=sys.stderr)
        print("trace recorded with the SAME binary and RF_LAUNCH_LOG set?",
              file=sys.stderr)
        return 1
    per_ms, per_n = {}, {}
    for name, dur in zip(names, merged):
        per_ms[name] = per_ms.get(name, 0.0) + dur / 1e6
        per_n[name] = per_n.get(name, 0) + 1
    total = sum(per_ms.values())
    print("total attributed GPU time %.1f ms" % total)
    print("%-28s %10s %8s %8s %10s" % ("site", "ms", "share", "n", "us/op"))
    for name in sorted(per_ms, key=lambda k: -per_ms[k]):
        print("%-28s %10.1f %7.1f%% %8d %10.1f"
              % (name, per_ms[name], 100 * per_ms[name] / total,
                 per_n[name], 1e3 * per_ms[name] / per_n[name]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("actions", nargs="+", choices=["record", "report", "attr"])
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--binary", default=BINARY)
    ap.add_argument("--launch-log", default=None,
                    help="path for the workload's launch log; required by "
                         "'attr', and passed to the workload during 'record'")
    # NOT default=["--profile"]: argparse APPENDS to a non-empty default,
    # so `--arg --profile-large` would have produced BOTH flags and the
    # binary honors --profile first -- the trace would silently be the
    # 100k one whatever was asked for.
    ap.add_argument("--arg", action="append", default=None,
                    help="argument for the traced binary (default: --profile)")
    a = ap.parse_args()
    if a.arg is None:
        a.arg = ["--profile"]
    for action in a.actions:
        if action == "record":
            rc = record(a.trace, a.binary, a.arg, launch_log=a.launch_log)
        elif action == "attr":
            if not a.launch_log:
                ap.error("'attr' needs --launch-log")
            rc = attr(a.trace, a.launch_log)
        else:
            rc = report(a.trace)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
