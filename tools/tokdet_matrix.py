#!/usr/bin/env python3
"""tools/tokdet_matrix.py -- drive the whole "are tokenizer vocabulary
trainers bitwise reproducible" matrix, one axis at a time, sequentially.

SHAPE OF THE EXPERIMENT. One BASE configuration, then one axis moved at a
time off that base, so a difference can be attributed:

  repeat   the base, N times. Plain run-to-run nondeterminism.
  threads  the base at 1, 2, 4, 8, 16 trainer threads.
  vocab    the base at a smaller and a larger vocabulary. Tie density moves
           with how many merges get taken, so a trainer can be reproducible
           at 1k and not at 32k.
  order    the base with the shard list reversed and rotated. Same bytes,
           different input order.

EVERY RUN IS A FRESH PROCESS. That is not tidiness. Rust's default HashMap
seeds its hasher per process, so two trains inside one interpreter could
agree for a reason that would not survive a real user running the trainer
twice. Separate processes is the honest test.

TWO CONTROLS RUN BEFORE ANY VERDICT IS BELIEVED.

  1. tokdet_compare.py --self-test, which perturbs a real trained vocabulary
     and asserts the comparison SEES each perturbation.
  2. An end-to-end control arm that trains two vocabularies that MUST differ
     (a vocabulary size off by one; for SentencePiece, also its documented
     shuffled-sample mode). If the pipeline reports those as identical, the
     pipeline is broken and every "identical" below is worthless.

  python3 tools/tokdet_matrix.py --corpus-dir DIR --out DIR --python PY \
      --threads 1 --reps 5
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def order_files(shards, how):
    if how == "fwd":
        return list(shards)
    if how == "rev":
        return list(reversed(shards))
    if how == "rot":
        return shards[1:] + shards[:1]
    raise ValueError(how)


def run_one(py, out_root, spec, env_extra, quiet=False):
    d = os.path.join(out_root, "runs", spec["id"])
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ)
    env.update(env_extra)
    if spec["kind"] == "hf":
        cmd = [py, os.path.join(HERE, "tokdet_train_hf.py"), "--out", d,
               "--vocab-size", str(spec["vocab"]), "--model", spec["model"],
               "--label", spec["id"], "--corpus"] + spec["files"]
    else:
        cmd = [py, os.path.join(HERE, "tokdet_train_sp.py"), "--out", d,
               "--vocab-size", str(spec["vocab"]), "--model-type", spec["model"],
               "--threads", str(spec["threads"]), "--label", spec["id"],
               "--shuffle", str(spec.get("shuffle", 0)),
               "--input-sentence-size", str(spec.get("iss", 0)),
               "--corpus"] + spec["files"]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    el = time.time() - t0
    if p.returncode != 0:
        return {"id": spec["id"], "dir": d, "ok": False, "elapsed": el,
                "stderr": p.stderr[-2000:]}
    if not quiet:
        print("  %-52s %6.1fs" % (spec["id"], el), flush=True)
    return {"id": spec["id"], "dir": d, "ok": True, "elapsed": el}


def env_for(kind, threads):
    if kind == "hf":
        return {"RAYON_NUM_THREADS": str(threads),
                "TOKENIZERS_PARALLELISM": "true" if threads > 1 else "false",
                "OMP_NUM_THREADS": str(threads)}
    return {"OMP_NUM_THREADS": str(threads)}


def compare_group(py, kind, sample, dirs, tag, out_root):
    j = os.path.join(out_root, "compare", tag + ".json")
    os.makedirs(os.path.dirname(j), exist_ok=True)
    cmd = [py, os.path.join(HERE, "tokdet_compare.py"), "--kind", kind,
           "--sample", sample, "--json", j, "--dirs"] + dirs
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {"tag": tag, "error": p.stderr[-2000:]}
    with open(j) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", default="1", help="comma list, e.g. 1,2,4,8,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--axis-reps", type=int, default=2)
    ap.add_argument("--base-vocab", type=int, default=8000)
    ap.add_argument("--vocabs", default="1000,32000")
    ap.add_argument("--orders", default="rev,rot")
    ap.add_argument("--kinds", default="hf,sp")
    ap.add_argument("--models", default="bpe")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.corpus_dir, "corpus_manifest.json")) as fh:
        man = json.load(fh)
    shards = [s["path"] for s in man["shards"]]
    sample = man["sample"]["path"]
    os.makedirs(args.out, exist_ok=True)

    threads = [int(x) for x in args.threads.split(",") if x]
    vocabs = [int(x) for x in args.vocabs.split(",") if x]
    orders = [x for x in args.orders.split(",") if x]
    kinds = [x for x in args.kinds.split(",") if x]
    models = [x for x in args.models.split(",") if x]
    base_t = threads[0]

    specs = []

    def add(kind, model, vocab, t, order, rep, arm, **kw):
        sid = "%s-%s-v%d-t%d-o%s-r%d" % (kind, model, vocab, t, order, rep)
        if kw.get("shuffle"):
            sid += "-shuf"
        specs.append(dict(id=sid, kind=kind, model=model, vocab=vocab, threads=t,
                          order=order, rep=rep, arm=arm,
                          files=order_files(shards, order), **kw))

    for kind, model in itertools.product(kinds, models):
        for r in range(args.reps):                                   # repeat
            add(kind, model, args.base_vocab, base_t, "fwd", r, "repeat")
        for t in threads[1:]:                                        # threads
            for r in range(args.axis_reps):
                add(kind, model, args.base_vocab, t, "fwd", r, "threads")
        for v in vocabs:                                             # vocab
            for r in range(args.axis_reps):
                add(kind, model, v, base_t, "fwd", r, "vocab")
        for o in orders:                                             # order
            for r in range(args.axis_reps):
                add(kind, model, args.base_vocab, base_t, o, r, "order")
        # end-to-end control: these MUST come back different
        add(kind, model, args.base_vocab + 1, base_t, "fwd", 0, "e2e_control")
        if kind == "sp":
            for r in range(2):
                add(kind, model, args.base_vocab, base_t, "fwd", r,
                    "e2e_control", shuffle=1, iss=200000)

    # dedupe (a vocab in --vocabs equal to base, etc.)
    seen, uniq = set(), []
    for s in specs:
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        uniq.append(s)
    specs = uniq

    print("%d runs planned" % len(specs), flush=True)
    runs = {}
    t_start = time.time()
    for s in specs:
        r = run_one(args.python, args.out, s, env_for(s["kind"], s["threads"]))
        r["spec"] = {k: v for k, v in s.items() if k != "files"}
        r["spec"]["files"] = [os.path.basename(f) for f in s["files"]]
        runs[s["id"]] = r
        if not r["ok"]:
            print("  FAILED %s: %s" % (s["id"], r["stderr"][-400:]), flush=True)

    # ---------------------------------------------------------- controls
    controls = {}
    if not args.skip_selftest:
        for kind in kinds:
            base_id = "%s-%s-v%d-t%d-ofwd-r0" % (kind, models[0], args.base_vocab, base_t)
            if base_id not in runs or not runs[base_id]["ok"]:
                continue
            j = os.path.join(args.out, "compare", "selftest-%s.json" % kind)
            os.makedirs(os.path.dirname(j), exist_ok=True)
            p = subprocess.run(
                [args.python, os.path.join(HERE, "tokdet_compare.py"), "--kind", kind,
                 "--sample", sample, "--self-test", "--dir", runs[base_id]["dir"],
                 "--tmp", os.path.join(args.out, "_selftest_" + kind), "--json", j],
                capture_output=True, text=True)
            with open(j) as fh:
                controls["selftest_" + kind] = json.load(fh)
            print("selftest %s: %s" % (kind, "PASS" if p.returncode == 0 else "FAIL"),
                  flush=True)

    # ------------------------------------------------------- comparisons
    cmps = []

    def group(kind, tag, ids):
        dirs = [runs[i]["dir"] for i in ids if i in runs and runs[i]["ok"]]
        if len(dirs) < 2:
            return
        res = compare_group(args.python, kind, sample, dirs, tag, args.out)
        res["tag"] = tag
        res["run_ids"] = [i for i in ids if i in runs and runs[i]["ok"]]
        cmps.append(res)

    for kind, model in itertools.product(kinds, models):
        b = "%s-%s-v%d" % (kind, model, args.base_vocab)
        group(kind, "%s__repeat" % b,
              ["%s-t%d-ofwd-r%d" % (b, base_t, r) for r in range(args.reps)])
        if len(threads) > 1:
            group(kind, "%s__threads" % b,
                  ["%s-t%d-ofwd-r0" % (b, base_t)]
                  + ["%s-t%d-ofwd-r0" % (b, t) for t in threads[1:]])
            for t in threads[1:]:
                group(kind, "%s__repeat_at_t%d" % (b, t),
                      ["%s-t%d-ofwd-r%d" % (b, t, r) for r in range(args.axis_reps)])
        group(kind, "%s__order" % b,
              ["%s-t%d-ofwd-r0" % (b, base_t)]
              + ["%s-t%d-o%s-r0" % (b, base_t, o) for o in orders])
        for o in orders:
            group(kind, "%s__repeat_at_o%s" % (b, o),
                  ["%s-t%d-o%s-r%d" % (b, base_t, o, r) for r in range(args.axis_reps)])
        for v in vocabs:
            vb = "%s-%s-v%d" % (kind, model, v)
            group(kind, "%s__repeat_at_v%d" % (vb, v),
                  ["%s-t%d-ofwd-r%d" % (vb, base_t, r) for r in range(args.axis_reps)])
        # end-to-end controls: MUST differ
        group(kind, "%s__E2E_CONTROL_vocab_plus_one" % b,
              ["%s-t%d-ofwd-r0" % (b, base_t),
               "%s-%s-v%d-t%d-ofwd-r0" % (kind, model, args.base_vocab + 1, base_t)])
        if kind == "sp":
            group(kind, "%s__E2E_CONTROL_shuffled_sample" % b,
                  ["%s-t%d-ofwd-r0-shuf" % (b, base_t),
                   "%s-t%d-ofwd-r1-shuf" % (b, base_t)])

    summary = {
        "corpus": man,
        "threads": threads, "reps": args.reps, "axis_reps": args.axis_reps,
        "base_vocab": args.base_vocab, "vocabs": vocabs, "orders": orders,
        "kinds": kinds, "models": models,
        "total_wall_sec": round(time.time() - t_start, 1),
        "runs": runs, "controls": controls, "comparisons": cmps,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n%-58s %s" % ("COMPARISON", "VERDICT"))
    for c in cmps:
        if "error" in c:
            print("%-58s ERROR" % c["tag"])
            continue
        v = "IDENTICAL" if c["all_identical"] else "DIFFERS"
        print("%-58s %s" % (c["tag"], v))
    print("\nsummary: %s" % os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
