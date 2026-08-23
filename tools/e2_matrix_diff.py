#!/usr/bin/env python3
"""E2 matrix differ: N machines' e2 directories -> one verdict table.

Each directory is the output of `tools/e2_matrix_fit.py` (and, when
present, `tools/e2_mojo_cards.sh`) on one machine. The first directory is
the REFERENCE column (Apple, by convention); every other column is judged
against it, per cell:

  IDENTICAL        prediction hash equal (and proba/model hashes where
                   present) AND the card has no differing stage
  OUTPUT-ONLY      prediction hash equal but the card differs at a named
                   stage -- an output identity WITHOUT a certificate
                   (E1's NVIDIA RMSE finding); reported as a divergence
  DIVERGENT@stage  first differing card stage (or "predictions" if the
                   cards agree but the outputs do not, which means a stage
                   is missing from the card)
  REFUSED=         both refused with the same message -- a pass
  REFUSED!=        refused with different messages, or one side refused
  MISSING          the cell exists on one side only
  ERROR/CRASHED    the cell failed on a side; the message is printed

Exit code is 0 only when every cell is IDENTICAL or REFUSED=. The table
is also written as e2_verdicts.md beside the first directory when
--write is given, for E2_RESULTS.md to include verbatim.

usage: python3 tools/e2_matrix_diff.py <ref_dir>:LABEL <dir>:LABEL [...] [--write]
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identity_trace_diff import TraceError, align, parse_trace  # noqa: E402


class CardError(Exception):
    pass


# The caller-visible output hashes a cell may carry. The first three are
# the tree matrix's; the rest are `tools/e2u_matrix_fit.py`'s (added
# 2026-08-23, additive). A key absent on both sides is not compared.
OUTPUT_KEYS = (
    "predictions", "proba", "model",
    "labels", "centroids", "inertia", "distances", "indices",
    "components", "explained_variance", "explained_variance_ratio",
    "singular_values", "mean", "noise_variance", "transformed",
    "coef", "intercept",
    "classes",  # KNeighborsClassifier.classes_ (e2u, 2026-08-23)
    "log_density",  # KernelDensity.score_samples (e2u, 2026-08-23)
)


def load_dir(path):
    cells = {}
    # `e2u_cells.json` is `tools/e2u_matrix_fit.py`'s record (the
    # unsupervised matrix, added 2026-08-23): same shape, same verdicts,
    # read the same way. Its cell names (`kmeans_*`, `knn_*`, `dbscan_*`,
    # `pca_*`, `tsvd_*`, `ols_*`) do not collide with the tree matrix's.
    for fn in ("e2_cells.json", "e2u_cells.json", "e1_fits.json",
               "e2_mojo_cards.json"):
        p = os.path.join(path, fn)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        block = d.get("cells") or d.get("fits") or d.get("cards") or {}
        for name, entry in block.items():
            if fn == "e1_fits.json" and name in cells:
                continue  # e2 carries the E1 four verbatim
            e = dict(entry)
            if "card" in e and e["card"]:
                e["card_path"] = os.path.join(path, e["card"])
            cells.setdefault(name, e)
    inputs = {}
    commit = None
    for fn in ("e2_cells.json", "e2u_cells.json"):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            inputs = d.get("inputs", {})
            commit = d.get("commit")
            break
    return cells, inputs, commit


import tempfile


def _last_fit(path):
    """A card file holding MORE than one fit (the trace APPENDS when two
    fits share a card path -- the E1 driver's four names collide with the
    E2 cells' in one directory) is split at the seq-0 record and the LAST
    fit is written to a temp file. Returns the path to parse."""
    with open(path) as fh:
        lines = fh.readlines()
    starts = [i for i, l in enumerate(lines)
              if l.split("\t")[0] == "0" and "\t" in l]
    if len(starts) <= 1:
        return path
    block = lines[starts[-1]:]
    # carry the comment header lines that precede the last block
    hdr = [l for l in lines[starts[-2]:starts[-1]] if l.startswith("#")]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".card", delete=False)
    tmp.writelines(hdr + block)
    tmp.close()
    return tmp.name


def first_card_divergence(path_a, path_b):
    """Returns (n_stages, first_differing_tag or None, note). A parse
    error raises CardError so the caller reports it as its OWN verdict --
    never as agreement."""
    try:
        ra, _, _ = parse_trace(_last_fit(path_a))
        rb, _, _ = parse_trace(_last_fit(path_b))
    except TraceError as exc:
        raise CardError(str(exc))
    al = align(ra, rb)
    note = ""
    if al.a_only or al.b_only:
        note = (f"tag sets differ (A-only {len(al.a_only)}, "
                f"B-only {len(al.b_only)})")
    for a, b in al.pairs:
        if a.hash != b.hash:
            return len(al.pairs), a.tag, note
    if al.a_only or al.b_only:
        first = (al.a_only or al.b_only)[0][1]
        return len(al.pairs), first, note
    return len(al.pairs), None, note


def judge(ref, other):
    if ref is None or other is None:
        return "MISSING", ""
    for key in ("crashed", "error"):
        if key in ref or key in other:
            side = "ref" if key in ref else "other"
            msg = (ref if key in ref else other).get(key)
            return key.upper(), f"{side}: {str(msg)[:90]}"
    if "refused" in ref or "refused" in other:
        if ref.get("refused") == other.get("refused"):
            return "REFUSED=", ref["refused"][:80]
        return "REFUSED!=", (f"ref={str(ref.get('refused'))[:50]} | "
                             f"other={str(other.get('refused'))[:50]}")
    out_same = all(ref.get(k) == other.get(k)
                   for k in OUTPUT_KEYS
                   if k in ref or k in other)
    pa, pb = ref.get("card_path"), other.get("card_path")
    if pa and pb and os.path.exists(pa) and os.path.exists(pb):
        try:
            n, tag, note = first_card_divergence(pa, pb)
        except CardError as exc:
            return "CARD-ERROR", str(exc)[:100]
        if tag is None and out_same:
            return "IDENTICAL", f"{n} stages" + (f" ({note})" if note else "")
        if tag is None and not out_same:
            which = [k for k in OUTPUT_KEYS
                     if ref.get(k) != other.get(k)]
            return "DIVERGENT@" + "+".join(which), f"cards agree on {n} stages"
        if out_same:
            return "OUTPUT-ONLY@" + tag, note
        return "DIVERGENT@" + tag, note
    if out_same:
        return "IDENTICAL(no-card)", ""
    which = [k for k in OUTPUT_KEYS
             if ref.get(k) != other.get(k)]
    return "DIVERGENT@" + "+".join(which), "no card"


def main():
    argv = [a for a in sys.argv[1:] if a != "--write"]
    write = "--write" in sys.argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    cols = []
    for a in argv:
        path, _, label = a.partition(":")
        cols.append((label or os.path.basename(path.rstrip("/")), path))
    loaded = [(label, *load_dir(path)) for label, path in cols]
    ref_label, ref_cells, ref_inputs, ref_commit = loaded[0]

    lines = []
    lines.append(f"# E2 verdicts, reference = {ref_label} @ {ref_commit}")
    lines.append("")
    for label, cells, inputs, commit in loaded[1:]:
        same_inputs = inputs == ref_inputs if inputs and ref_inputs else None
        lines.append(f"- {label}: commit {commit}, inputs "
                     f"{'IDENTICAL' if same_inputs else 'DIFFER' if same_inputs is False else 'n/a'}"
                     f" ({len(cells)} cells)")
    lines.append("")
    names = sorted(set().union(*[set(c[1]) for c in loaded]))
    head = "| cell | " + " | ".join(l for l, *_ in loaded[1:]) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (len(loaded)))
    rc = 0
    tally = {}
    for name in names:
        row = [name]
        for label, cells, _, _ in loaded[1:]:
            v, note = judge(ref_cells.get(name), cells.get(name))
            tally.setdefault(label, {}).setdefault(v.split("@")[0], 0)
            tally[label][v.split("@")[0]] += 1
            if v not in ("IDENTICAL", "REFUSED=", "IDENTICAL(no-card)"):
                rc = 1
            row.append(f"**{v}**" + (f" ({note})" if note else "")
                       if v not in ("IDENTICAL", "REFUSED=") else
                       f"{v}" + (f" ({note})" if note and v == "IDENTICAL" else ""))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for label, t in tally.items():
        lines.append(f"- {label}: " + ", ".join(
            f"{k} {n}" for k, n in sorted(t.items())))
    text = "\n".join(lines)
    print(text)
    if write:
        out = os.path.join(cols[0][1], "e2_verdicts.md")
        with open(out, "w") as fh:
            fh.write(text + "\n")
        print("wrote", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
