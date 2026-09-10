#!/usr/bin/env python3
"""Witness for the Sep 9 vs Sep 10 rf-clf/et-clf `predict` divergence: same
fixtures and models as tools/identity_break.py; hash today's predict as-is and
re-cast to the label dtype (int32) and compare both to the Sep 9 part hashes."""
import json, os, sys
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
sys.path.insert(0, "/root/mojolearn/tools"); sys.path.insert(0, "/root/mojolearn/python")
import numpy as np
import identity_break as ib
import mojolearn as ml
sep9 = json.load(open("/root/trees_out/ib/sep9_h100_baseline.json"))["cells"]
ok = bad = 0
print(f"mode={ml.numeric_mode()}")
for lane, cls in (("rf-clf", ml.RandomForestClassifier), ("et-clf", ml.ExtraTreesClassifier)):
    for f in ib.FIXTURES:
        X, yc, yr = ib.fixture(f)
        m = cls(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
        pred = m.predict(X); proba = m.predict_proba(X)
        pa = np.asarray(pred)
        h_today = ib._h(pred); h_cast = ib._h(pa.astype(yc.dtype)); h_proba = ib._h(proba)
        ref = sep9[f"{lane}/{f}"]["parts"][0]
        verdict = "EXPLAINED" if (h_cast == ref["predict"] and h_proba == ref["proba"]) else "UNEXPLAINED"
        if verdict == "EXPLAINED": ok += 1
        else: bad += 1
        print(f"{lane}/{f}: today predict dtype={pa.dtype} shape={pa.shape} hash={h_today} | cast to {yc.dtype} hash={h_cast} sep9 predict={ref['predict']} | proba {h_proba} sep9 {ref['proba']} -> {verdict}")
print(f"summary: explained={ok} unexplained={bad}")
