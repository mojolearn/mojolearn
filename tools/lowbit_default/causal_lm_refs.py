#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""lane/lowbit-default: the verifier's causal-LM lanes against the reference
table, from an identity_break JSON (`--lanes hf-causal-lm`): every hashed
part of every cell equal to the table's `ref`. The lanes load fp32_v1 by
name, so this holds under any process default. Exit 1 on any difference or
when nothing was compared."""
import json
import sys

run = json.load(open(sys.argv[1]))
table = json.load(open(sys.argv[2]))["cells"]
seen = bad = 0
for key, c in sorted(run["cells"].items()):
    ref = table.get(key)
    if ref is None:
        continue
    for part in ("train", "infer", "batch"):
        want = (ref.get(part) or {}).get("ref")
        if not want or want.startswith("n/a"):
            continue
        vals = c.get("hashes") if part == "train" else c.get(part)
        if not vals:
            print(f"{key} {part}: NOT HASHED ({c.get('verdict')}, {c.get(part + '_verdict')})")
            bad += 1
            continue
        ok = all(str(v).startswith(want) or want.startswith(str(v)) for v in vals)
        seen += 1
        bad += 0 if ok else 1
        print(f"{key} {part}: {'EQUAL' if ok else 'DIFFERENT'} ref {want} got {sorted(set(map(str, vals)))}")
print(f"compared {seen} parts, {bad} not equal")
sys.exit(0 if seen and not bad else 1)
