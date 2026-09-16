#!/usr/bin/env python3
"""tools/tokdet_compare.py -- compare trained vocabulary artifacts, in three
layers, and PROVE the comparison can fail before anyone trusts it.

THE THREE LAYERS, and why three.

  bytes      sha256 of every artifact file. Answers "are these the same
             bytes". Normalizes nothing.
  struct     parse the artifact and compare the vocabulary as a structure:
             token -> id map, and the merge list as a SEQUENCE and as a SET.
             Sequence-vs-set is the whole point: it separates "the same
             merges written in a different order" from "different merges".
  tokenize   run held-out text through both and compare the id sequence AND
             the piece sequence. Ids differ but pieces match => a renumbering,
             the same tokenization spelled differently. Pieces differ => the
             vocabularies genuinely disagree about how text splits.

WHY --self-test EXISTS. A byte comparison that quietly normalizes, or a
load-then-compare that ignores ordering, reports "identical" for everything
and teaches us nothing. So --self-test runs BOTH controls on real artifacts:

  POSITIVE: an exact copy must come back IDENTICAL on all three layers. A
            comparison that always screams is as useless as one that never
            does.
  NEGATIVE: several one-at-a-time perturbations, each of a DIFFERENT kind,
            each asserted to fire the specific layer it should fire. If a
            perturbation does not trip its layer, this exits non-zero and
            says so. Watch it fail on the perturbed side before believing any
            "identical" it reports later.

  python3 tools/tokdet_compare.py --kind hf --sample F --dirs A B [C ...]
  python3 tools/tokdet_compare.py --kind hf --sample F --self-test --dir A
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(d):
    out = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and name != "record.json":
            out[name] = sha256_file(p)
    return out


# ----------------------------------------------------------------- loaders


def load_hf(d):
    with open(os.path.join(d, "tokenizer.json"), "rb") as fh:
        blob = json.load(fh)
    model = blob["model"]
    vocab = dict(model.get("vocab", {}))
    merges = []
    for m in model.get("merges", []):
        merges.append(tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split(" ", 1)))
    return {"vocab": vocab, "merges": merges}


def load_sp(d):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "sp.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    scores = [sp.get_score(i) for i in range(sp.get_piece_size())]
    vocab = {p: i for i, p in enumerate(pieces)}
    return {"vocab": vocab, "pieces": pieces, "scores": scores, "merges": list(enumerate(pieces))}


def load_struct(kind, d):
    return load_hf(d) if kind == "hf" else load_sp(d)


def encode(kind, d, lines):
    if kind == "hf":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        encs = [tok.encode(ln) for ln in lines]
        return [e.ids for e in encs], [e.tokens for e in encs]
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "sp.model"))
    ids = [sp.encode(ln) for ln in lines]
    toks = [[sp.id_to_piece(i) for i in row] for row in ids]
    return ids, toks


# --------------------------------------------------------------- comparers


def cmp_bytes(a, b):
    ha, hb = artifact_hashes(a), artifact_hashes(b)
    names = sorted(set(ha) | set(hb))
    per = {n: {"a": ha.get(n), "b": hb.get(n), "same": ha.get(n) == hb.get(n)} for n in names}
    return {"identical": all(v["same"] for v in per.values()), "files": per}


def cmp_struct(kind, a, b):
    sa, sb = load_struct(kind, a), load_struct(kind, b)
    va, vb = sa["vocab"], sb["vocab"]
    only_a = sorted(set(va) - set(vb))
    only_b = sorted(set(vb) - set(va))
    shared = set(va) & set(vb)
    id_diff = sorted(t for t in shared if va[t] != vb[t])
    ma, mb = sa["merges"], sb["merges"]
    first_div = None
    for i in range(min(len(ma), len(mb))):
        if ma[i] != mb[i]:
            first_div = i
            break
    if first_div is None and len(ma) != len(mb):
        first_div = min(len(ma), len(mb))
    out = {
        "n_tokens_a": len(va),
        "n_tokens_b": len(vb),
        "vocab_same_set": not only_a and not only_b,
        "vocab_same_map": not only_a and not only_b and not id_diff,
        "n_tokens_only_in_a": len(only_a),
        "n_tokens_only_in_b": len(only_b),
        "n_tokens_id_differs": len(id_diff),
        "example_only_in_a": only_a[:8],
        "example_only_in_b": only_b[:8],
        "example_id_differs": [
            {"token": t, "a": va[t], "b": vb[t]} for t in id_diff[:8]
        ],
        "n_merges_a": len(ma),
        "n_merges_b": len(mb),
        "merges_same_sequence": ma == mb,
        "merges_same_set": set(ma) == set(mb),
        "first_divergent_merge_index": first_div,
        "example_divergent_merges": [
            {"index": i, "a": list(ma[i]) if i < len(ma) else None,
             "b": list(mb[i]) if i < len(mb) else None}
            for i in range(first_div, min(first_div + 5, max(len(ma), len(mb))))
        ] if first_div is not None else [],
    }
    if kind == "sp":
        n = min(len(sa["scores"]), len(sb["scores"]))
        sd = [i for i in range(n) if sa["scores"][i] != sb["scores"][i]]
        out["n_scores_differ"] = len(sd)
        out["max_abs_score_delta"] = max(
            (abs(sa["scores"][i] - sb["scores"][i]) for i in sd), default=0.0
        )
        out["example_scores_differ"] = [
            {"id": i, "piece": sa["pieces"][i], "a": sa["scores"][i], "b": sb["scores"][i]}
            for i in sd[:8]
        ]
    out["identical"] = bool(out["vocab_same_map"] and out["merges_same_sequence"]
                            and out.get("n_scores_differ", 0) == 0)
    return out


def cmp_tokenize(kind, a, b, lines):
    ia, ta = encode(kind, a, lines)
    ib, tb = encode(kind, b, lines)
    ids_diff = [i for i in range(len(lines)) if ia[i] != ib[i]]
    tok_diff = [i for i in range(len(lines)) if ta[i] != tb[i]]
    ex = []
    for i in tok_diff[:5]:
        ex.append({"line": i, "a": ta[i][:24], "b": tb[i][:24]})
    return {
        "n_lines": len(lines),
        "n_lines_ids_differ": len(ids_diff),
        "n_lines_pieces_differ": len(tok_diff),
        "identical": not ids_diff and not tok_diff,
        "same_pieces_different_ids": bool(ids_diff and not tok_diff),
        "total_tokens_a": sum(len(x) for x in ta),
        "total_tokens_b": sum(len(x) for x in tb),
        "example_piece_differences": ex,
    }


def compare(kind, a, b, lines):
    r = {"a": a, "b": b,
         "bytes": cmp_bytes(a, b),
         "struct": cmp_struct(kind, a, b),
         "tokenize": cmp_tokenize(kind, a, b, lines)}
    r["identical"] = all(r[k]["identical"] for k in ("bytes", "struct", "tokenize"))
    return r


def read_sample(path, n):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    return [ln for ln in lines if ln][:n]


# --------------------------------------------------------------- self-test


def perturb_hf(src, dst, how):
    shutil.copytree(src, dst)
    p = os.path.join(dst, "tokenizer.json")
    with open(p) as fh:
        blob = json.load(fh)
    model = blob["model"]
    if how == "swap_two_vocab_ids":
        toks = sorted(model["vocab"], key=lambda t: model["vocab"][t])
        x, y = toks[-1], toks[-2]
        model["vocab"][x], model["vocab"][y] = model["vocab"][y], model["vocab"][x]
    elif how == "reorder_two_merges":
        m = model["merges"]
        m[-1], m[-2] = m[-2], m[-1]
    elif how == "drop_one_merge":
        m = model["merges"]
        dropped = m.pop(len(m) // 2)
        tok = ("".join(dropped) if isinstance(dropped, (list, tuple))
               else dropped.replace(" ", ""))
        model["vocab"].pop(tok, None)
    else:
        raise ValueError(how)
    with open(p, "w") as fh:
        json.dump(blob, fh)
    return dst


def perturb_sp(src, dst, how):
    from sentencepiece import sentencepiece_model_pb2 as pb2

    shutil.copytree(src, dst)
    p = os.path.join(dst, "sp.model")
    proto = pb2.ModelProto()
    with open(p, "rb") as fh:
        proto.ParseFromString(fh.read())
    if how == "bump_one_score":
        proto.pieces[len(proto.pieces) // 2].score -= 1.0
    elif how == "swap_two_pieces":
        i, j = len(proto.pieces) - 1, len(proto.pieces) - 2
        a = copy.deepcopy(proto.pieces[i])
        b = copy.deepcopy(proto.pieces[j])
        proto.pieces[i].CopyFrom(b)
        proto.pieces[j].CopyFrom(a)
    elif how == "drop_one_piece":
        # remove a long, mid-vocabulary piece so held-out text must respell it
        idx = max(range(3, len(proto.pieces)), key=lambda k: len(proto.pieces[k].piece))
        del proto.pieces[idx]
    else:
        raise ValueError(how)
    with open(p, "wb") as fh:
        fh.write(proto.SerializeToString())
    return dst


# each control names the layers that MUST fire; anything else is reported but
# not asserted, because "did the tokenization move" is a finding, not a given.
CONTROLS = {
    "hf": [
        ("swap_two_vocab_ids", ["bytes", "struct"], "vocab ids renumbered"),
        ("reorder_two_merges", ["bytes", "struct"], "same merges, different order"),
        ("drop_one_merge", ["bytes", "struct", "tokenize"], "a merge genuinely gone"),
    ],
    "sp": [
        ("bump_one_score", ["bytes", "struct"], "one unigram score moved"),
        ("swap_two_pieces", ["bytes", "struct"], "same pieces, different ids"),
        ("drop_one_piece", ["bytes", "struct", "tokenize"], "a piece genuinely gone"),
    ],
}


def self_test(kind, d, lines, tmp):
    results = {"kind": kind, "subject": d, "controls": [], "failures": []}

    # POSITIVE CONTROL -------------------------------------------------
    pos = os.path.join(tmp, "copy_identical")
    if os.path.exists(pos):
        shutil.rmtree(pos)
    shutil.copytree(d, pos)
    r = compare(kind, d, pos, lines)
    ok = r["identical"]
    results["controls"].append(
        {"control": "POSITIVE exact copy", "expect": "identical on all layers",
         "layers_fired": [k for k in ("bytes", "struct", "tokenize") if not r[k]["identical"]],
         "passed": ok}
    )
    if not ok:
        results["failures"].append(
            "POSITIVE control failed: an exact copy did not compare identical "
            "(layers that fired: %s). The comparison is broken."
            % [k for k in ("bytes", "struct", "tokenize") if not r[k]["identical"]]
        )

    # NEGATIVE CONTROLS ------------------------------------------------
    for how, must_fire, why in CONTROLS[kind]:
        dst = os.path.join(tmp, "perturbed_" + how)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        (perturb_hf if kind == "hf" else perturb_sp)(d, dst, how)
        r = compare(kind, d, dst, lines)
        fired = [k for k in ("bytes", "struct", "tokenize") if not r[k]["identical"]]
        missed = [k for k in must_fire if k not in fired]
        results["controls"].append(
            {"control": "NEGATIVE " + how, "means": why, "must_fire": must_fire,
             "layers_fired": fired, "passed": not missed,
             "struct": {k: r["struct"][k] for k in
                        ("vocab_same_map", "merges_same_sequence", "merges_same_set",
                         "n_tokens_id_differs", "n_scores_differ") if k in r["struct"]},
             "tokenize": {k: r["tokenize"][k] for k in
                          ("n_lines_ids_differ", "n_lines_pieces_differ",
                           "same_pieces_different_ids")}}
        )
        if missed:
            results["failures"].append(
                "NEGATIVE control '%s' (%s) did NOT trip layer(s) %s. A comparison "
                "that cannot see this perturbation cannot be trusted to see a real "
                "difference either." % (how, why, missed)
            )
    results["passed"] = not results["failures"]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("hf", "sp"), required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--sample-lines", type=int, default=400)
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--json")
    args = ap.parse_args()

    lines = read_sample(args.sample, args.sample_lines)

    if args.self_test:
        tmp = args.tmp or os.path.join(os.path.dirname(args.dir.rstrip("/")), "_selftest")
        os.makedirs(tmp, exist_ok=True)
        out = self_test(args.kind, args.dir, lines, tmp)
    else:
        base = args.dirs[0]
        out = {"kind": args.kind, "base": base, "sample_lines": len(lines),
               "pairs": [compare(args.kind, base, d, lines) for d in args.dirs[1:]]}
        out["all_identical"] = all(p["identical"] for p in out["pairs"])

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    if args.self_test and not out["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
