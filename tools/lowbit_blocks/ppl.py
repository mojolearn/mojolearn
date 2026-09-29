#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-blocks (e): the model's held-out perplexity under a profile,
on lane/lowbit-quality's two texts, by that lane's protocol.

    # 1. the ids, with an interpreter that has `tokenizers` (Lane B's venv):
    python tools/lowbit_blocks/ppl.py ids --corpus .../enwik8/input.txt --tail-from 99000000 \
        --tokenizer /root/models/SmolLM2-360M/tokenizer.json --out ids_enwik8.bin
    # 2. the score, with the package's interpreter (numpy + mojolearn):
    python tools/lowbit_blocks/ppl.py score --ids ids_enwik8.bin --model ... --profile fixed15_v1 --out <json>

THE PROTOCOL IS LANE B's (`bench/lowbit_quality/infer_eval.py` on
lane/lowbit-quality, `evaluation_ids` and `score`): the tail from the first
byte after the first newline at or after `--tail-from`, strict UTF-8 (an
invalid byte is one U+FFFD), the model's own tokenizer.json, no special token;
the first 200 x 512 ids in non-overlapping windows; every position but the
first of a window scored from the ids before it in the same window; the nll
in float64 from the float32 logits (log-softmax in float64). The ids' sha256
must equal the one lane/lowbit-quality recorded, or the texts differ.

The gate is opened in THIS process only, as in `model_logits.py`.
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def cmd_ids(a):
    from tokenizers import Tokenizer
    raw = open(a.corpus, "rb").read()
    nl = raw.find(b"\n", a.tail_from)
    start = nl + 1
    end = raw.rfind(b"\n") + 1
    exact = raw[start:end].decode("utf-8", errors="surrogateescape")
    text = "".join("�" if 0xDC80 <= ord(ch) <= 0xDCFF else ch for ch in exact)
    enc = Tokenizer.from_file(a.tokenizer).encode(text, add_special_tokens=False)
    need = a.windows * a.length
    used = enc.ids[:need]
    if len(used) < need:
        raise SystemExit(f"only {len(used)} ids")
    b = b"".join(int(t).to_bytes(4, "little", signed=True) for t in used)
    open(a.out, "wb").write(b)
    rec = {"corpus": a.corpus, "tail_from": a.tail_from, "byte_start": start, "windows": a.windows,
           "length": a.length, "ids_sha256": hashlib.sha256(b).hexdigest()}
    json.dump(rec, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(rec))


def cmd_score(a):
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
    from mojolearn import _numeric_profile as NP
    if a.profile == "fixed15_v1":
        NP.PROFILES["fixed15_v1"]["inference"] = True  # THIS PROCESS ONLY
    from mojolearn.models import CausalLM
    from mojolearn._array import Array
    from mojolearn._bufcheck import flat_view
    raw = open(a.ids, "rb").read()
    ids = np.frombuffer(raw, dtype="<i4").reshape(-1, a.length)
    if a.limit:
        ids = ids[:a.limit]
    lm = CausalLM.load(a.model, numeric_profile=a.profile, device=a.device)
    v = lm.vocab_size
    nll_w = []
    top_all = []
    t0 = time.perf_counter()
    for lo in range(0, ids.shape[0], a.batch):
        chunk = ids[lo:lo + a.batch]
        lg = lm.forward(Array.from_list(chunk.tolist(), "<i4"))
        x = np.frombuffer(bytes(flat_view(lg, "f").cast("B")), dtype="<f4").reshape(chunk.shape[0], a.length, v)
        for r in range(chunk.shape[0]):
            rows = x[r, :-1].astype(np.float64)
            m = rows.max(axis=1, keepdims=True)
            lse = np.log(np.exp(rows - m).sum(axis=1)) + m[:, 0]
            tgt = chunk[r, 1:]
            nll = lse - rows[np.arange(a.length - 1), tgt]
            nll_w.append(nll)
            top_all.append(x[r, :-1].argmax(axis=1))
        print(f"windows {lo + chunk.shape[0]}/{ids.shape[0]} {time.perf_counter() - t0:.0f} s", flush=True)
    nll = np.stack(nll_w)
    mean = float(nll.mean())
    rec = {"profile": lm.numeric_profile, "device": lm.device, "windows": int(ids.shape[0]),
           "ids_sha256": hashlib.sha256(raw).hexdigest(), "mean_nll": mean, "perplexity": float(np.exp(mean)),
           "window_mean_nll": nll.mean(axis=1).tolist(),
           "top1_sha256": hashlib.sha256(np.concatenate(top_all).astype("<i4").tobytes()).hexdigest(),
           "nll_sha256": hashlib.sha256(nll.astype("<f8").tobytes()).hexdigest()}
    json.dump(rec, open(a.out, "w"), indent=1)
    np.save(a.out + ".top1.npy", np.stack(top_all).astype("<i4"))
    print(f"RESULT ppl profile={lm.numeric_profile} text={os.path.basename(a.ids)} mean_nll={mean:.12f} "
          f"perplexity={rec['perplexity']:.9f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ids")
    i.add_argument("--corpus", required=True)
    i.add_argument("--tail-from", type=int, required=True)
    i.add_argument("--tokenizer", required=True)
    i.add_argument("--windows", type=int, default=200)
    i.add_argument("--length", type=int, default=512)
    i.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--ids", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--profile", default="fp32_v1")
    s.add_argument("--device", default="auto")
    s.add_argument("--length", type=int, default=512)
    s.add_argument("--batch", type=int, default=4)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_ids(a) if a.cmd == "ids" else cmd_score(a)


if __name__ == "__main__":
    main()
