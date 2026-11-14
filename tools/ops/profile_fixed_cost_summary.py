#!/usr/bin/env python3
"""Summarize one race's py-spy collapsed stacks (or a cProfile pstats file) for fixed-cost attribution.

Collapsed line: ``frame;frame;...;leaf N``. The public call is the first frame matching --under (default: the
mojolearn public methods); samples whose stack holds no such frame (data load, upload, the harness) are reported
as one line and otherwise ignored. Reported: total samples, samples under the call, then the top frames under
the call by self samples and by inclusive samples (native frames from the Mojo binding .so appear by symbol).
"""
import argparse, collections, os, re, sys

DEFAULT_UNDER = r"(fit|predict|predict_proba|transform|score_samples|forward|decode|generate|run_program|run|step|train_step|kneighbors)\b.*mojolearn|mojolearn.*(\.fit|\.predict|\.transform|\.forward|\.run|\.step|\.generate|\.kneighbors)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True)
    ap.add_argument("--collapsed")
    ap.add_argument("--pstats")
    ap.add_argument("--under", default=DEFAULT_UNDER)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = ["race %s" % a.race]
    if a.collapsed and os.path.exists(a.collapsed) and os.path.getsize(a.collapsed) > 0:
        under = re.compile(a.under)
        total = 0; inside = 0; selfc = collections.Counter(); incl = collections.Counter(); outside_leaf = collections.Counter()
        for raw in open(a.collapsed, errors="replace"):
            raw = raw.rstrip("\n")
            if not raw or " " not in raw:
                continue
            stack, n = raw.rsplit(" ", 1)
            try:
                n = int(n)
            except ValueError:
                continue
            frames = stack.split(";")
            total += n
            start = next((i for i, f in enumerate(frames) if under.search(f)), None)
            if start is None:
                outside_leaf[frames[-1][:120]] += n
                continue
            inside += n
            sub = frames[start:]
            selfc[sub[-1][:140]] += n
            for f in dict.fromkeys(sub):
                incl[f[:140]] += n
        lines.append("samples total=%d under_call=%d (%.1f%%) outside=%d" % (total, inside, 100.0 * inside / max(total, 1), total - inside))
        lines.append("-- top self frames under the call")
        for f, n in selfc.most_common(a.top):
            lines.append("%7d %5.1f%%  %s" % (n, 100.0 * n / max(inside, 1), f))
        lines.append("-- top inclusive frames under the call")
        for f, n in incl.most_common(a.top):
            lines.append("%7d %5.1f%%  %s" % (n, 100.0 * n / max(inside, 1), f))
        lines.append("-- top leaves outside the call (data load, harness)")
        for f, n in outside_leaf.most_common(8):
            lines.append("%7d  %s" % (n, f))
    elif a.pstats and os.path.exists(a.pstats):
        import pstats, io
        s = io.StringIO(); st = pstats.Stats(a.pstats, stream=s); st.sort_stats("tottime").print_stats(a.top)
        lines.append("cProfile (Python frames only; native time is inside the binding call)"); lines.append(s.getvalue())
    else:
        lines.append("no profile data")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
