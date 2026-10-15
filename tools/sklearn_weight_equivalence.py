# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""scikit-learn's check_sample_weight_equivalence_on_dense_data on every public
estimator whose fit takes sample_weight, outside the expected-failure list, with
the failure text (how far apart the two fits are). Test tooling.

    python tools/sklearn_weight_equivalence.py <out.tsv>
"""
import inspect
import sys
import warnings

from sklearn.utils.estimator_checks import check_sample_weight_equivalence_on_dense_data as CHECK

from mojolearn.tests import test_sklearn_compat as T

warnings.simplefilter("ignore")
rows = []
for name in sorted(T.SHARED_NAMES):
    try:
        est = T._make(name)
    except Exception:
        continue
    if "sample_weight" not in inspect.signature(est.fit).parameters:
        continue
    try:
        CHECK(name, est)
        status, msg = "pass", ""
    except Exception as e:
        status, msg = "FAIL", (type(e).__name__ + ": " + str(e)).replace("\n", " ").replace("\t", " ")[:400]
    rows.append((name, status, msg))
    print(f"{status:5s} {name} {msg[:160]}", flush=True)
with open(sys.argv[1], "w") as f:
    f.write("estimator\tstatus\tdetail\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print("EQUIV total=%d fail=%d" % (len(rows), sum(r[1] == "FAIL" for r in rows)))
