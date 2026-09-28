"""Tabulate one tools/ann_apple2_ab.sh output (lane ann-apple3, 2026-09-28).

    python3 tools/ann_apple3_tab.py <speed.stdout> [--stages] [--quality]

Prints, per cell and tier, every arm's seconds (fit, first search, second
search; one value per run, the stage pass last), whether every arm's digests
are equal, and the digests. `--stages` adds the stage pass's marks summed per
(driver, stage) and cluster/'s k-means marks summed per stage; `--quality`
the ANN-QUALITY rows per arm. Reads text only; runs nothing.
"""
import collections
import json
import re
import sys


def main():
    path = sys.argv[1]
    want_stages = "--stages" in sys.argv
    want_quality = "--quality" in sys.argv
    arms = []
    cells = collections.OrderedDict()      # (algo, mode) -> arm -> [dict]
    stages = collections.OrderedDict()     # (mode, arm) -> (driver, stage) -> [ms, count]
    km = collections.OrderedDict()         # (mode, arm) -> stage -> [ms, count]
    quality = collections.OrderedDict()    # (data, algo) -> arm -> [row]
    line_re = re.compile(r"^\[(\S+) (\S+)\] (.*)$")
    for raw in open(path, errors="replace"):
        line = raw.rstrip("\n")
        if line.startswith("AB arms:"):
            for tok in line.split(" (after=")[0].split()[2:]:
                arms.append(tok.split("=")[0])
            continue
        m = line_re.match(line)
        if not m:
            if line.startswith(("BUILD FAIL", "ARMS NOT BUILT", "[build ")):
                print(line)
            continue
        arm, mode, body = m.groups()
        if body.startswith("{"):
            try:
                d = json.loads(body)
            except ValueError:
                continue
            for algo, v in d.items():
                cells.setdefault((algo, mode), collections.OrderedDict()).setdefault(arm, []).append(v)
        elif body.startswith("ANN-STAGE "):
            p = body.split()
            if len(p) >= 4:
                slot = stages.setdefault((mode, arm), collections.OrderedDict()).setdefault((p[1], p[2]), [0.0, 0])
                slot[0] += float(p[3])
                slot[1] += 1
        elif body.startswith("KMSTAGE "):
            p = body.split()
            name = p[1]
            slot = km.setdefault((mode, arm), collections.OrderedDict()).setdefault(name, [0.0, 0])
            slot[0] += float(p[-1])
            slot[1] += 1
        elif body.startswith("ANN-QUALITY "):
            d = json.loads(body[len("ANN-QUALITY "):])
            quality.setdefault((d["data"], d["algo"]), collections.OrderedDict()).setdefault(arm, []).append(d)
        elif "Traceback" in body or "Error" in body:
            print(line)

    def fmt(rows, key, places=3):
        vals = [r.get(key) for r in rows]
        if all(v is None for v in vals):
            return None
        return "/".join("-" if v is None else f"{v:.{places}f}" for v in vals)

    for (algo, mode), by_arm in cells.items():
        digests = {arm: sorted({(r.get("model"), r.get("out"), r.get("out2")) for r in rows})
                   for arm, rows in by_arm.items()}
        same = len({json.dumps(v) for v in digests.values()}) == 1
        print(f"## {algo} {mode}: digests {'EQUAL in every arm' if same else 'DIFFER BETWEEN ARMS'}")
        for arm, rows in by_arm.items():
            parts = []
            for key, places in (("fit_s", 3), ("search_s", 3), ("search2_s", 4), ("refine_s", 3), ("refine2_s", 4)):
                got = fmt(rows, key, places)
                if got is not None:
                    parts.append(f"{key} {got}")
            warm = [r for r in rows if r.get("out2") is not None and r.get("out2") != r.get("out")]
            note = "  SECOND SEARCH DIGEST DIFFERS" if warm else ""
            print(f"  {arm}: " + "  ".join(parts) + f"  {digests[arm]}{note}")
    if want_stages:
        for (mode, arm), table in stages.items():
            print(f"## stages {mode} {arm} (ms summed over the stage pass, count)")
            for (driver, stage), (ms, count) in table.items():
                print(f"  {driver} {stage} {ms:.1f} ({count})")
        for (mode, arm), table in km.items():
            print(f"## k-means marks {mode} {arm} (ms summed, count)")
            for name, (ms, count) in table.items():
                print(f"  {name} {ms:.1f} ({count})")
    if want_quality:
        for (data, algo), by_arm in quality.items():
            print(f"## quality {data} {algo}")
            for arm, rows in by_arm.items():
                if "recall" in rows[0]:
                    vals = [r["recall"] for r in rows]
                    print(f"  {arm}: recall " + " ".join(f"{v:.4f}" for v in vals) + f"  mean {sum(vals) / len(vals):.4f}")
                elif "trust10" in rows[0]:
                    t = [r["trust10"] for r in rows]
                    kl = [r["kl"] for r in rows]
                    print(f"  {arm}: trust10 " + " ".join(f"{v:.4f}" for v in t) + f"  mean {sum(t) / len(t):.4f}"
                          + "  kl " + " ".join(f"{v:.4f}" for v in kl) + f"  mean {sum(kl) / len(kl):.4f}")


if __name__ == "__main__":
    main()
