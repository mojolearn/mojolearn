#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-default: the RESIDENT sessions under fixed15_v1 against the
per-layer route, bit for bit, on SmolLM2-360M (B=2, L=64 ids of
tools/lowbit_blocks/model_logits.py).

For each profile (the default, fixed15_v1, and fp32_v1 by name):
  prefill   the full logits through every block's resident
            `TransformerDecodeSession.forward` (L=64), the embedding, norm
            and head as `CausalLM._run` spells them: sha256 must equal
            `CausalLM.forward`'s (fixed15_v1 d37c2ea81d13743a..., fp32_v1
            833c9a8947bdd619...)
  decode    resident sessions: a prefill of p in {1, 7} tokens then one
            `step` per token; every position's logits against the full
            prefill's, bit for bit
  generate  `CausalLM.generate` from a 512-token prompt, N new tokens:
            the resident route (`_generate_resident`, which must RUN) against
            the per-layer route; the ids equal, and the resident session's
            last-pass logits equal the per-layer forward's at that position
Exit 1 on any mismatch. The same script on a tree with a sabotage patch
applied must exit 1 (its fixed15_v1 comparisons move).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "lowbit_blocks"))

WANT = {"fixed15_v1": "d37c2ea81d13743a", "fp32_v1": "833c9a8947bdd619"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--box", default=os.uname().nodename)
    ap.add_argument("--experimental-hip", action="store_true")
    args = ap.parse_args()
    from model_logits import token_ids, raw, sha, SEED
    from mojolearn.models import CausalLM
    if args.experimental_hip:
        original = CausalLM._generate_resident
        def experimental(self, *a, **kw):
            return original(self, *a, **kw, _experimental_int15_resident=True)
        CausalLM._generate_resident = experimental
    from mojolearn._array import Array
    from mojolearn._linalg_impl import matmul_int15
    from mojolearn._transformer_impl import TransformerDecodeSession
    bad = []

    def check(name, ok):
        print(f"RESULT check {name}: {'HELD' if ok else 'BROKEN'}", flush=True)
        if not ok:
            bad.append(name)

    for kw in ({}, {"numeric_profile": "fixed15_v1"}, {"numeric_profile": "fp32_v1"}):
        lm = CausalLM.load(args.model, **kw)
        prof = lm.numeric_profile
        v, d = lm.vocab_size, lm.d_model
        print(f"== {prof} ({'no keyword' if not kw else 'named'}), device {lm.device}", flush=True)
        rows = token_ids(2, 64, v)
        ids = Array.from_list(rows, "<i4")
        full = raw(lm.forward(ids))
        check(f"{prof} per-layer hash {WANT[prof]}", sha(full).startswith(WANT[prof]))

        def through(sessions, chunk, step):
            b, l = len(chunk), len(chunk[0])
            n = b * l
            x = lm._prims.embedding(lm._embed, Array.from_list(chunk, "<i4").reshape((n,))).reshape((b, l, d))
            for ss in sessions:
                x = ss.step(x) if step else ss.forward(x)
            hn = lm._prims.rms_norm(x.reshape((n, d)), lm._norm, lm.norm_eps)
            out = matmul_int15(hn, lm._head_int15) if lm._head_int15 is not None else lm._prims.linear(hn, lm._head)
            return raw(out)

        def opened(b, cap):
            st = lm.allocate_state(b, cap)
            return st, [TransformerDecodeSession(blk, s) for blk, s in zip(lm._blocks, st.layers)]

        st, ss = opened(2, 64)
        try:
            pre = through(ss, rows, False)
        finally:
            for s in ss:
                s.close()
        check(f"{prof} resident prefill == per-layer ({sha(pre)[:16]})", pre == full)
        row = 64 * v * 4
        for p in (1, 7):
            st, ss = opened(1, 64)
            try:
                got = [through(ss, [rows[0][:p]], False)]
                for t in range(p, 64):
                    got.append(through(ss, [[rows[0][t]]], True))
            finally:
                for s in ss:
                    s.close()
            cat = b"".join(got)
            diff = sum(1 for t in range(64) if cat[t * v * 4:(t + 1) * v * 4] != full[t * v * 4:(t + 1) * v * 4])
            check(f"{prof} resident decode prefix {p}: {diff} of 64 positions differ", diff == 0 and len(cat) == row)

        prompt = Array.from_list(token_ids(1, 512, v, SEED + 7), "<i4")
        n = args.new
        from mojolearn._buffer import empty
        last = empty((1, v), "<f4")
        res = lm._generate_resident(prompt, n, 512 + n, last_logits=last)
        check(f"{prof} the resident route ran", res is not None)
        inner = lm._generate_resident
        lm._generate_resident = lambda *a, **k: None
        try:
            per = lm.generate(prompt, n)
        finally:
            lm._generate_resident = inner
        if res is not None:
            check(f"{prof} resident generate ids == per-layer ({sha(res.tobytes())[:16]})",
                  res.tobytes() == per.tobytes())
            seq = Array.from_list([per.tolist()[0][:512 + n - 1]], "<i4")
            want = raw(lm.forward(seq))[(512 + n - 2) * v * 4:]
            check(f"{prof} resident last-pass logits == per-layer forward at position {512 + n - 2}",
                  raw(last) == want)
        del lm
    print("RESIDENT GATE " + ("GREEN" if not bad else "RED: " + "; ".join(bad)), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
