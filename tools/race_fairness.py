#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RACE FAIRNESS, applied by code (AGENTS.md "Race fairness: what the measurements are for", Andrew 2026-10-13).

The boards measure the cost of bitwise identity; every result is shown, wins and losses alike, and only a broken
comparison is hidden. This module is the one place the rule runs. It marks every (mode, phase) row of every race
of a board.json, and tools/main_board_ingest.py calls it on every rebuild. The paper (mlsys) and mojolearn.org
read the marks and omit rows with rerace "x" and opponent arms in race["opponents_excluded"].

    python3 tools/race_fairness.py bench/results/bench_board/main-nvidia-l40s/board.json ...   # apply, write
    python3 tools/race_fairness.py --check <board.json>...    # print what would change, write nothing
    python3 tools/race_fairness.py --review <board.json>...   # list warning signs on rows still shown

What it applies, in order:
 1. Opponent arms that are not opponents are excluded from the race (race["opponents_excluded"]), and the race is
    judged against its remaining opponents: code our harness wrote (HARNESS_ARMS) and torch arms measured on a
    host shown to run identical work measurably slower (SLOW_HOSTS).
 2. A row a human audit judged (race["audit"][key]["audit"] set) keeps that verdict while the measurements it
    judged are unchanged (its "pairing" fingerprint); UNFAIR hides it, FAIR/SUSPECT/TIE show it.
 3. Every other row is judged by fairness_rule(): different iteration counts on the two arms -> UNFAIR (hidden);
    otherwise MATCHED (settings read back and equal), TIE or UNCHECKED, all shown.
Nothing here edits a measurement.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

TIE_BAND = (0.95, 1.05)

# 1a. Code our harness wrote rather than a library's own implementation: never an opponent (Andrew, 2026-10-13).
# lane -> predicate on the arm name. (OLS torch-gpu is torch.linalg.lstsq, a library routine, and stays.)
_TORCH = lambda arm: str(arm).startswith("torch")
HARNESS_ARMS = {
    "kmeans": lambda a: a == "torch-gpu",          # k-means from torch ops (index_add_)
    "knn": lambda a: a == "torch-gpu",             # cdist + topk
    "pca": lambda a: a == "torch-gpu",             # covariance + torch.linalg.eigh assembled by the harness
    "ols": lambda a: a == "torch-gpu-eigh",        # normal equations by eigh, harness-written
}
for _lane in ("lm-forward", "lm-train-step", "lm-infer", "lm-host-train-step", "transformer-forward",
              "transformer-infer", "mamba1-forward", "mamba2-forward", "mamba3-forward", "mamba1-infer",
              "mamba2-infer", "mamba3-infer", "samba-forward", "samba-train-step", "samba-infer", "moe",
              "resnet-block"):
    HARNESS_ARMS[_lane] = _TORCH                   # harness twins and pure-PyTorch references (the stored cells)

# 1b. Hosts shown to run identical work measurably slower than another box of the same GPU model. A torch arm
# measured there is excluded, except on the GPU-bound linear-algebra lanes below: the slowness was measured on small
# neural workloads where host overhead dominates, and torch's linear-algebra times on the host reproduce elsewhere
# (randomized-svd 66.1 vs 65.7 ms). Other libraries' cells from the host are not judged.
GPU_BOUND_LANES = {"cholesky", "eigh", "lstsq", "lu-factor", "lu-solve", "qr", "svd", "svdvals", "randomized-svd",
                   "ols", "pca", "gemm", "gemm-bf16"}
SLOW_HOSTS = {
    "cc560ebdaf91": "L40S host about 2x slower than host 24a11adce16e on identical torch work "
                    "(lstm-clf eager-fp32 1246 vs 516 ms; fairness audit 2026-10-13)",
}


def phase_cells(race, phase):
    return (race.get("cells") if phase == "fit" else race.get("infer_cells")) or []


def completed(cell):
    ms = cell.get("median_ms")
    return cell.get("status") == "ok" and isinstance(ms, (int, float)) and ms > 0


def our_arm(mode):
    return "ours-fast" if mode == "fast" else "ours"


def n_iter(cell):
    q = (cell or {}).get("quality") or {}
    for k in ("n_iter", "n_iter_", "iterations", "n_steps"):
        if isinstance(q.get(k), (int, float)) and not isinstance(q.get(k), bool):
            return q[k]
    return None


def cell_host(cell):
    host = ((cell.get("copied_from") or {}).get("box")) or ""
    return host or (cell.get("hardware") or "")


def pairing(race, mode, phase):
    """Fingerprint of what one (mode, phase) row measures: our arm's and every opponent's status and median."""
    rows = sorted((str(c.get("arm")), str(c.get("status")),
                   round(c["median_ms"], 6) if isinstance(c.get("median_ms"), (int, float)) else None)
                  for c in phase_cells(race, phase)
                  if c.get("arm") == our_arm(mode) or c.get("mode") == "opponent")
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def exclusions(race):
    """{arm: reason} for the opponent arms rules 1a and 1b exclude from this race."""
    out = {}
    pred = HARNESS_ARMS.get(race.get("lane"))
    for c in phase_cells(race, "fit") + phase_cells(race, "infer"):
        arm = c.get("arm")
        if c.get("mode") != "opponent" or arm in out:
            continue
        if pred and pred(arm):
            out[arm] = "code our harness wrote, not a library's own implementation"
        elif str(c.get("library")) == "torch" and race.get("lane") not in GPU_BOUND_LANES:
            for host, why in SLOW_HOSTS.items():
                if host in cell_host(c):
                    out[arm] = "measured on %s: %s" % (host, why)
    return out


def exclude(race, today):
    """Merge rule 1 into race["opponents_excluded"] (arms a human audit excluded stay)."""
    found = exclusions(race)
    if not found:
        return
    ex = race.setdefault("opponents_excluded", {"arms": [], "reason": "", "date": today})
    why = ex.setdefault("why", {})
    for arm, reason in sorted(found.items()):
        if arm not in ex["arms"]:
            ex["arms"] = sorted(ex["arms"] + [arm])
        why.setdefault(arm, reason)
    if not ex.get("reason"):
        ex["reason"] = "; ".join("%s: %s" % kv for kv in sorted(why.items()))


def fairness_rule(race, mode, phase, today):
    """The entry for a row no audit covers, or None when there is no row (no our cell, or no completed opponent that
    is not excluded)."""
    cells = phase_cells(race, phase)
    ours = next((c for c in cells if c.get("arm") == our_arm(mode)), None)
    excluded = set((race.get("opponents_excluded") or {}).get("arms") or [])
    opps = [c for c in cells if c.get("mode") == "opponent" and completed(c) and c.get("arm") not in excluded]
    if ours is None or not opps:
        return None
    best = min([c for c in opps if c.get("device") == "gpu"] or opps, key=lambda c: c["median_ms"])
    ratio = ours["median_ms"] / best["median_ms"] if completed(ours) else None
    entry = {"rule": "race_fairness.fairness_rule (2026-10-13)", "date": today, "opponent": best.get("arm"),
             "ours_over_theirs": round(ratio, 3) if ratio is not None else None,
             "pairing": pairing(race, mode, phase)}
    differ = [(c.get("arm"), n_iter(c)) for c in opps
              if n_iter(ours) is not None and n_iter(c) is not None and n_iter(c) != n_iter(ours)]
    if differ:
        # Either arm running more iterations than the other is a broken comparison.
        entry.update(verdict="UNFAIR", rerace="x",
                     reason="iterations differ: ours %s, %s" % (n_iter(ours), ", ".join("%s %s" % d for d in differ)))
    elif race.get("params_check") != "MATCHED":
        entry.update(verdict="UNCHECKED", rerace="",
                     reason="no MATCHED settings-parity check of both arms and no audit yet; shown until audited")
    elif ratio is not None and TIE_BAND[0] <= ratio <= TIE_BAND[1]:
        entry.update(verdict="TIE", rerace="", reason="a tie (ours/theirs within TIE_BAND); shown")
    else:
        entry.update(verdict="MATCHED", rerace="", reason="settings-parity check MATCHED")
    return entry


def apply(board, prev=None, today=None):
    """Mark every row of every race of `board` (rules 1-3). `prev`: the previous version of the board, whose audit
    entries and exclusions carry over while unchanged (None: the board's own). Returns the races marked "x"."""
    today = today or datetime.date.today().isoformat()
    prev = board if prev is None else prev
    modes = (board.get("config") or {}).get("modes") or ["identical"]
    marked = 0
    for section in ("races", "extra_races"):
        for rid, rr in ((board.get(section) or {}).items()):
            prev_rr = (((prev or {}).get("races") or {}).get(rid)
                       or ((prev or {}).get("extra_races") or {}).get(rid) or {})
            old = prev_rr.get("audit") or {}
            if prev_rr.get("opponents_excluded") and not rr.get("opponents_excluded"):
                rr["opponents_excluded"] = prev_rr["opponents_excluded"]
            exclude(rr, today)
            audit = {}
            for mode in modes:
                for phase in ("fit", "infer"):
                    key = "%s/%s" % (mode, phase)
                    entry = old.get(key)
                    keep = entry and entry.get("pairing") == pairing(rr, mode, phase) and not entry.get("rule")
                    entry = entry if keep else fairness_rule(rr, mode, phase, today)
                    if entry:
                        audit[key] = entry
            if audit:
                rr["audit"] = audit
            else:
                rr.pop("audit", None)
            if any(e.get("rerace") == "x" for e in audit.values()):
                rr["rerace"] = "x"
                marked += 1
            else:
                rr.pop("rerace", None)
    return marked


def review(board):
    """Warning signs on rows still shown, for a human to judge: (lane, dataset, key, signs)."""
    out = []
    for section in ("races", "extra_races"):
        for rr in ((board.get(section) or {}).values()):
            excluded = set((rr.get("opponents_excluded") or {}).get("arms") or [])
            for key, e in sorted((rr.get("audit") or {}).items()):
                if e.get("rerace") == "x":
                    continue
                mode, phase = key.split("/")
                cells = phase_cells(rr, phase)
                ours = next((c for c in cells if c.get("arm") == our_arm(mode)), None)
                opp = next((c for c in cells if c.get("arm") == e.get("opponent")), None)
                if ours is None or opp is None or opp.get("arm") in excluded:
                    continue
                signs = []
                r = e.get("ours_over_theirs")
                if isinstance(r, (int, float)) and (r < 0.2 or r > 20):
                    signs.append("extreme ratio %.3g" % r)
                qo, qt = ours.get("quality") or {}, opp.get("quality") or {}
                for k in ("accuracy", "r2", "auc", "logloss", "inertia", "explained_variance_ratio_sum", "mean_log_likelihood"):
                    a, b = qo.get(k), qt.get(k)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and max(abs(a), abs(b)) > 0:
                        if abs(a - b) / max(abs(a), abs(b)) > 0.05:
                            signs.append("quality %s %.4g vs %.4g" % (k, a, b))
                        break
                times = ours.get("times_ms") or []
                med = ours.get("median_ms")
                if times and isinstance(med, (int, float)) and all(abs(t - med) > 1e-6 * max(1, med) for t in times):
                    signs.append("our median is not among our timed rounds")
                if e.get("verdict") == "UNCHECKED":
                    signs.append("no audit")
                if signs:
                    out.append((rr.get("lane"), rr.get("dataset"), key, e.get("verdict"), signs))
    return out


def _load(path):
    raw = open(path).read()
    board = json.loads(raw)
    second = raw.split("\n")[1] if "\n" in raw else ""
    fmt = dict(indent=len(second) - len(second.lstrip()) or None, tail="\n" if raw.endswith("\n") else "",
               sort=json.dumps(board, indent=len(second) - len(second.lstrip()) or None, sort_keys=True,
                               default=str) + ("\n" if raw.endswith("\n") else "") == raw)
    return board, raw, fmt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boards", nargs="+")
    ap.add_argument("--check", action="store_true", help="print what would change; write nothing")
    ap.add_argument("--review", action="store_true", help="list warning signs on rows still shown")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    args = ap.parse_args(argv)
    for path in args.boards:
        board, raw, fmt = _load(path)
        if args.review:
            rows = review(board)
            print("%s: %d shown rows with warning signs" % (path, len(rows)))
            for lane, ds, key, verdict, signs in rows:
                print("  %-22s %-14s %-15s %-9s %s" % (lane, ds, key, verdict, "; ".join(signs)))
            continue
        before = json.loads(raw)
        marked = apply(board, None, args.date)
        changed = []
        for section in ("races", "extra_races"):
            for rid, rr in (board.get(section) or {}).items():
                old = (before.get(section) or {}).get(rid) or {}
                for key, e in (rr.get("audit") or {}).items():
                    o = (old.get("audit") or {}).get(key) or {}
                    if (o.get("verdict"), o.get("rerace")) != (e.get("verdict"), e.get("rerace")):
                        changed.append((rr.get("lane"), rr.get("dataset"), key, o.get("verdict"), e.get("verdict")))
                if (old.get("opponents_excluded") or {}).get("arms") != (rr.get("opponents_excluded") or {}).get("arms"):
                    changed.append((rr.get("lane"), rr.get("dataset"), "excluded arms",
                                    (old.get("opponents_excluded") or {}).get("arms"),
                                    (rr.get("opponents_excluded") or {}).get("arms")))
        print("%s: %d races marked x, %d changes" % (path, marked, len(changed)))
        for ch in changed[:60]:
            print("  %-22s %-14s %-15s %s -> %s" % ch)
        if not args.check:
            text = json.dumps(board, indent=fmt["indent"], sort_keys=fmt["sort"], default=str) + fmt["tail"]
            with open(path + ".tmp", "w") as fh:
                fh.write(text)
            os.replace(path + ".tmp", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
