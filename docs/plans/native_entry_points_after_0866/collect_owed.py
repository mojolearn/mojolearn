#!/usr/bin/env python3
"""Collect every row the performance fix lanes left open into owed.tsv.

Reads the lane tables (p3-*, perf2-*, perf3-*, pass2-*) and the third ideas
pass (ideas-pass3-from-scratch.tsv) in the evidence directory, keeps the rows
whose status says OWED, OWED-SHARED, partly / partial, or LEFT because a new
native entry is needed, merges rows that describe the same site, removes the
rows listed in dropped.tsv (closed by a later lane; checked in the code), and
assigns each remaining row a workstream (rules below, overrides.tsv on top).

Usage: python3 collect_owed.py [EVIDENCE_DIR]   (default ~/mojolearn-evidence)
Writes owed.tsv and counts.txt next to this script. Tool script, stdlib only.
"""
import csv, glob, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EVID = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/mojolearn-evidence")
PATTERNS = ["p3-*.md", "perf2-*.md", "perf3-*.md", "pass2-*.md"]
ROUND_ORDER = {"perf2": 0, "perf3": 1, "pass2": 2, "p3": 3}

NEW_ENTRY = re.compile(
    r"new (native |provider |worker |device |kernel )?(entry|op\b|operation|binding|protocol|kernel|abi|owner|door)"
    r"|needs? (a |an |one )?(new|native|device|signature|provider|kernel|abi)"
    r"|signature change|entry point|abi change|device-resident|resident on the device|stay(s)? on the device",
    re.I)
FILE_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:mojo|py|sh|json|toml))(?::([0-9][0-9,\- ]*))?")


def lane_of(path):
    base = os.path.basename(path)[:-3]
    rnd = base.split("-", 1)[0]
    lane = base
    # some files are named <round>-<round>-<lane>.md
    if lane.startswith(rnd + "-" + rnd + "-"):
        lane = lane[len(rnd) + 1:]
    return rnd, lane


def cells_of(line):
    # split on unescaped pipes outside backticks
    out, cur, tick = [], [], False
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|"); i += 2; continue
        if c == "`":
            tick = not tick
        if c == "|" and not tick:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur).strip())
    return out


def status_class(text):
    if re.search(r"OWED-SHARED", text, re.I):
        return "OWED-SHARED"
    if re.search(r"\bOWED\b", text):
        return "OWED"
    if re.search(r"\bowed\b", text):
        return "OWED"
    if re.search(r"\bpartly\b|\bpartial\b", text, re.I):
        return "PARTLY"
    if re.search(r"\bLEFT\b", text) and NEW_ENTRY.search(text):
        return "LEFT-NEW-ENTRY"
    return ""


def parse_table_file(path):
    rnd, lane = lane_of(path)
    rows, header = [], None
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    for n, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            header = None if not line.strip() == "" else header
            continue
        c = cells_of(line)
        if all(re.fullmatch(r":?-+:?", x or "-") for x in c):
            continue
        low = [x.lower() for x in c]
        if any(x in ("status", "verdict", "state", "outcome") for x in low) and not re.match(r"\d+$", c[0]):
            header = low
            continue
        rank, site, what, status = "", "", "", ""
        if header:
            def col(*names):
                for i, h in enumerate(header):
                    if any(h.startswith(nm) for nm in names) and i < len(c):
                        return c[i]
                return ""
            rank = col("rank")
            num = col("#")
            site = col("site", "file")
            what = col("what", "form")
            si = max(i for i, h in enumerate(header) if h in ("status", "verdict", "state", "outcome"))
            status = " ".join(c[si:]) if si < len(c) else ""
            local = rank or num
        else:
            if rnd == "p3" and re.match(r"\d+$", c[0]) and len(c) >= 3:
                rank, site, status = c[0], c[1], " ".join(c[2:])
                local = rank
            else:
                continue
        cls = status_class(status)
        if not cls:
            continue
        rows.append(dict(round=rnd, lane=lane, table=os.path.basename(path), line=n, rank=rank if rnd == "p3" else "",
                         local=local, site=site.replace("`", ""), what=what.replace("`", ""),
                         status=status.replace("`", ""), cls=cls))
    return rows


def site_files(site):
    """[(file, set(lines))] named in a site cell."""
    out = []
    for m in FILE_RE.finditer(site):
        f = m.group(1)
        if "/" not in f:
            f = "bindings/" + f if f.endswith(".mojo") else f
        ls = set()
        for part in re.split(r"[ ,]+", m.group(2) or ""):
            if not part:
                continue
            a, _, b = part.partition("-")
            try:
                a = int(a); b = int(b) if b else a
            except ValueError:
                continue
            ls.update(range(a, min(b, a + 400) + 1))
        out.append((f, ls))
    return out


def load_pass3():
    by_rank, by_file = {}, defaultdict(list)
    p = os.path.join(EVID, "ideas-pass3-from-scratch.tsv")
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            by_rank[r["rank"]] = r
            for f, ls in site_files(r["site"]):
                by_file[f].append((ls, r))
    return by_rank, by_file


def guess_kind_freq(text):
    t = text.lower()
    if re.search(r"compile|import of|unproven form", t):
        k = "6"
    elif re.search(r"stale|invalidat|cache hazard|leak|never dropped|generation stamp", t):
        k = "5"
    elif re.search(r"per step|per epoch|training loop|fit loop|per round|per boosting|per iteration|per member|per fold|per candidate", t):
        k = "4"
    elif re.search(r"upload|download|round-trip|copy|copies|twice|re-read|refit|re-validat|recomput|host", t):
        k = "3"
    elif re.search(r"per (row|item|element|label|token|document|sample|tree|feature|class)", t):
        k = "1"
    else:
        k = "2"
    if re.search(r"per (row|item|element|label|token|document|sample|tree|feature|step|batch)", t):
        f = "per-item"
    elif re.search(r"per (fit|epoch|fold|round|member|candidate)|fit loop|at fit|\bfit\b", t):
        f = "per-fit"
    elif re.search(r"once per process|cold|rare|refusal|error path", t):
        f = "rare"
    else:
        f = "per-call"
    return k, f


def load_tsv(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [r for r in csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t")]


# Workstream rules. File-path rules run first (the file names the owner),
# then wording rules; first match wins. overrides.tsv (source_table,
# workstream) wins over all; dropped.tsv removes a row.
FILE_RULES = [
    ("K-tools-gate", r"tools/|\.github/|build[a-z_]*\.sh|merged_build"),
    ("I-py310-311", r"watcher|3\.10|3\.11|class_watch_ready"),
    ("D-generation-sessions", r"generation|decode|parallel_causal|causal_lm|causal_operation|_run_engine|neural_prefill|"
                              r"mamba_public|transformer_decode|kv_|sampler|causal_model_config|inference_session"),
    ("C-training-sessions", r"byte_lm|parallel_byte|samba|_training\.|training/|optimizer|autograd|parallel_neural|"
                            r"parallel_mlp|cnn|sequence/|sequence_|xseq|_x_sequence|tsa|arima|neural_block|"
                            r"_mojolearn_transformer|embedding|residual_dropout|mlp_fit|recurrent|lm_corpus|tokenizer|"
                            r"transformer_binding|neural_descriptor|moe"),
    ("F-kernel-call-table", r"mani_|manifold|linalg_ops_kit|_mojolearn_metrics(_host)?\.mojo"),
    ("B-device-rows", r"msel|cross_val|fold|resample|bagging|adaboost|stacking|voting|rfe|iterative|trees_public_meta|"
                      r"trees_meta|multioutput|calibrat|trees_bagging|boost_admission|_native_msel_dispatch"),
    ("H-labels-metrics", r"label|metric|class_weights|classical_pandas|prep_labels|scorer|_expansion_metrics|"
                         r"_expansion_neighbors|_expansion_trees|_forest_host|_gbdt_host|classical_native\.mojo"),
    ("E-admitted-call", r"cpython_table|cpython\b|native_loaded_module|native_class_admission|native_prepared_owner|"
                        r"public_attributes|params_metadata|mode_metadata|estimator_binding|side_records|_records|"
                        r"admission|sequential_admission|family_raw_attr|native_identity_key|estimator_lifecycle|"
                        r"public_lifecycle|lifecycle|family_public|arena_array_metadata|classical_attributes|"
                        r"classical_validate|prep_owner|prep_classify_prog|arena_prog|arena_materialize|array_native|"
                        r"host_sparse|native_scalar_decode|classical_archive|prep_transform_pyapi|prep_encode|"
                        r"scaler_lifecycle|preprocessing\.py|density\.py|_hierarchy_impl|_sk_validate|_mode\.py|"
                        r"msel_common"),
    ("G-fused-entries", r"decomp_kit|classical_tsqr|classical_ridge|classical_ols|pls_|tsvd|ipca|pca_|dict_learning|"
                        r"linalg_decomp|als_native|kpca|mcd_|rp_native|x_decomp|_mojolearn_linalg|_mojolearn_estimators|"
                        r"logistic_native|host_logistic|x_linear|gp_|_mojolearn_gp|svc_|host_svc|_mojolearn_svm|"
                        r"svm_|iforest|neighbors_local|neighbors_semi|mcls_cluster|_mojolearn_hdbscan|cfb_spectral|"
                        r"mixture|kernel_methods|hotpath_device|x_prep_native_owner|prep_classify_native|minmax"),
    ("A-resident-state", r"_mojolearn_rf|_mojolearn_trees|_mojolearn_forest|trees_public|host_tree|host_iforest|"
                         r"resident|cluster_native|_mojolearn_x_cluster|kmeans|cfb_|neighbors_knn|neighbors_ann|"
                         r"_mojolearn_resample|_mojolearn_mixture|apple_fast"),
]
WORD_RULES = [
    ("J-multi-device-and-gaps", r"multioutputclassifier\.score|multi-device|several devices|x-decomp-lda|x-decomp-ipca|shard"),
    ("P-profile-first", r"profile|unknown|not traced|unexplained|reading owed"),
    ("D-generation-sessions", r"generat|decode|greedy|kv state|engine output"),
    ("C-training-sessions", r"train_step|trainer|epoch|multi-step|training session"),
    ("B-device-rows", r"fold|member|devicerows|device rows|meta-estimator"),
    ("G-fused-entries", r"fused|whole-step|one native call|one entry|single entry|one crossing|predict-all|predict_all"),
    ("H-labels-metrics", r"label|metric|classes_"),
    ("E-admitted-call", r"admi|record|dlsym|nativemodules|estimator_arguments|stamp|import|getattr|mro"),
    ("A-resident-state", r"resident|upload|download|copy|copies|handle|pointer|refit"),
]
WS_BY_LANE = {
    "trees": "A-resident-state", "neighbors-cluster": "A-resident-state", "cluster": "A-resident-state",
    "neighbors": "A-resident-state", "gp-svm": "G-fused-entries", "svm-gp-kernel": "G-fused-entries",
    "linear": "G-fused-entries", "pca-decomp": "G-fused-entries", "decomp-manifold-mixture": "G-fused-entries",
    "labels": "H-labels-metrics", "labels-metrics": "H-labels-metrics", "label-doors": "H-labels-metrics",
    "prep": "E-admitted-call", "prep-metrics": "H-labels-metrics", "msel": "B-device-rows",
    "msel-parallel": "B-device-rows", "lm-tokenizer": "D-generation-sessions", "neural": "C-training-sessions",
    "neural-blocks": "C-training-sessions", "samba-optim-autograd": "C-training-sessions",
    "cnn-sequence-tsa": "C-training-sessions", "seq-neural": "C-training-sessions",
    "gate-tools": "K-tools-gate", "forms-cleanup": "E-admitted-call", "shared-table": "E-admitted-call",
    "shared": "E-admitted-call", "core-cpython": "E-admitted-call", "array-arena": "E-admitted-call",
    "nbr-msel-trees": "B-device-rows", "cluster-gp-gate": "A-resident-state", "prep-shared": "E-admitted-call",
    "seq-trees-decomp": "A-resident-state", "cluster-nbr": "A-resident-state",
}


def assign_ws(row):
    files = row["file_line"].lower()
    for ws, rx in FILE_RULES:
        if re.search(rx, files):
            return ws
    text = " | ".join([row["what"], row["status_text"], row["fix"]]).lower()
    for ws, rx in WORD_RULES:
        if re.search(rx, text):
            return ws
    return WS_BY_LANE.get(row["lane"], "E-admitted-call")


def main():
    by_rank, by_file = load_pass3()
    raw = []
    for pat in PATTERNS:
        for path in sorted(glob.glob(os.path.join(EVID, pat))):
            raw.extend(parse_table_file(path))

    rows = []
    for r in raw:
        p3 = by_rank.get(r["rank"]) if r["rank"] else None
        src = "pass3-rank"
        if not p3:
            # join an earlier-round row to the third ideas pass by file and line overlap
            best, score = None, 0
            for f, ls in site_files(r["site"]):
                for pls, cand in by_file.get(f, []):
                    s = len(ls & pls) * 10 + 1 if (ls and pls and ls & pls) else 0
                    if s > score:
                        best, score = cand, s
            p3 = best
            src = "pass3-overlap" if best else "guess"
        if p3:
            kind, freq, methods = p3["kind"], p3["frequency"], p3["public_methods"]
            what = p3["what"] if src == "pass3-rank" or not r["what"] else r["what"]
            fix = p3["fix"]
        else:
            kind, freq = guess_kind_freq(r["what"] + " " + r["status"])
            methods, what, fix = "", r["what"], ""
        rows.append(dict(
            source_table="%s:%d" % (r["table"], r["line"]), round=r["round"], lane=r["lane"], rank=r["rank"],
            pass3_rank=(p3["rank"] if p3 else ""), status=r["cls"], file_line=r["site"], kind=kind, frequency=freq,
            kind_source=src, public_methods=methods, what=what, status_text=r["status"], fix=fix))

    # de-duplicate: same pass3 row reached from two tables, or same file with overlapping lines
    rows.sort(key=lambda x: (-ROUND_ORDER[x["round"]], x["source_table"]))
    kept, seen_rank, merged = [], {}, 0
    for r in rows:
        key = r["pass3_rank"] if r["kind_source"] != "guess" else ""
        if key and key in seen_rank:
            seen_rank[key]["also_in"].append(r["source_table"]); merged += 1
            continue
        r["also_in"] = []
        if key:
            seen_rank[key] = r
        kept.append(r)

    # code_checks.tsv: every OWED-SHARED row and every earlier-round row was
    # checked against integ/perf-20261105 (CLOSED / PARTLY / OPEN / NOT-CODE
    # with a file:line). CLOSED rows are dropped and listed in dropped.tsv.
    checks = {c["source_table"]: c for c in load_tsv("code_checks.tsv")}
    overrides = {o["source_table"]: o["workstream"] for o in load_tsv("overrides.tsv")}
    final, drop_out = [], []
    for r in kept:
        hits = [checks[s] for s in [r["source_table"]] + r["also_in"] if s in checks]
        closed = [c for c in hits if c["verdict"] == "CLOSED"]
        if closed and all(c["verdict"] == "CLOSED" for c in hits):
            drop_out.append((r, closed[0])); continue
        r["code_check"] = "; ".join("%s: %s" % (c["verdict"], c["evidence"]) for c in hits)
        r["workstream"] = overrides.get(r["source_table"]) or assign_ws(r)
        if hits and all(c["verdict"] == "NOT-CODE" for c in hits) and not r["workstream"].startswith("K"):
            r["workstream"] = "P-profile-first"
        final.append(r)
    with open(os.path.join(HERE, "dropped.tsv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["source_table", "file_line", "closed_by"])
        for r, c in sorted(drop_out, key=lambda x: x[0]["source_table"]):
            w.writerow([r["source_table"], r["file_line"], c["evidence"]])

    order = {"per-item": 0, "per-call": 1, "per-fit": 2, "rare": 3}
    final.sort(key=lambda x: (x["workstream"], order.get(x["frequency"], 9), x["kind"], x["source_table"]))
    cols = ["id", "source_table", "lane", "file_line", "kind", "frequency", "public_methods", "what", "proposed_fix",
            "workstream", "status", "pass3_rank", "kind_source", "also_in", "code_check"]
    with open(os.path.join(HERE, "owed.tsv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for i, r in enumerate(final, 1):
            r["id"] = "N%03d" % i
            fix = r["fix"]
            st = re.sub(r"\s+", " ", r["status_text"])
            proposed = (st + (" || ideas pass 3: " + fix if fix else "")).strip()
            w.writerow([r["id"], r["source_table"], r["lane"], r["file_line"], r["kind"], r["frequency"],
                        r["public_methods"], re.sub(r"\s+", " ", r["what"]), proposed, r["workstream"], r["status"],
                        r["pass3_rank"], r["kind_source"], ",".join(r["also_in"]), r.get("code_check", "")])

    with open(os.path.join(HERE, "counts.txt"), "w", encoding="utf-8") as fh:
        def p(*a):
            fh.write(" ".join(str(x) for x in a) + "\n")
        p("rows with an open status in the lane tables:", len(raw))
        p("merged as the same site:", merged)
        p("dropped as closed by a later lane or the compile fixes (dropped.tsv):", len(drop_out))
        p("owed rows kept:", len(final))
        for name, key in (("status", "status"), ("kind", "kind"), ("frequency", "frequency"), ("round", "round"),
                          ("workstream", "workstream"), ("kind_source", "kind_source")):
            p("by " + name + ":")
            for k, v in sorted(Counter(r[key] for r in final).items()):
                p("  ", k, v)
        p("workstream x frequency:")
        ct = Counter((r["workstream"], r["frequency"]) for r in final)
        for ws in sorted({r["workstream"] for r in final}):
            p("  ", ws, " ".join("%s=%d" % (f, ct[(ws, f)]) for f in order))
    print(open(os.path.join(HERE, "counts.txt")).read())


if __name__ == "__main__":
    main()
