# Classical host gate, lane `logistic-multiclass`, Apple M4, 2026-09-14

LogisticRegression with three classes (the softmax loss, lane/logistic-multiclass)
fitted on the Metal path and predicted through the CPU binding
`python/mojolearn/host/_mojolearn_estimators_host.so`, compared bit for bit
over `tools/identity_break.py`'s nine fixtures. Box: one Apple M4 (macOS
26.5.2, arm64), the same machine for both halves; commit bba81880d for the
recordings (the host binding was built from the working tree of the commit
that adds this directory).

    tools/classical_host_gate.py record  bench/results/classical_host/2026-09-14-apple-m4-multiclass --lanes logistic-multiclass
    tools/classical_host_gate.py check   bench/results/classical_host/2026-09-14-apple-m4-multiclass --report check_apple-m4_host.json
    MOJOLEARN_HOST_DIR=<set built with -D MOJOLEARN_HOST_SABOTAGE=1> MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
    tools/classical_host_gate.py check   ... --expect-mismatch --report check_apple-m4_sabotage.json

Each fixture directory holds `model.npz` (the fitted three-class model, its
`classes` member carrying the class count), `fixture.json` (the held-out
rows' hash) and `expected.json` (the SHA-256, dtype and shape of
`predict_proba`, `predict` and `decision_function` on the Metal path, the
reload byte-equal).

`check_apple-m4_host.json`: verdict IDENTICAL, 9 fixtures, every surface
EQUAL (36 comparisons: predict_proba, predict, decision_function and the
identity_break hash per fixture). No GPU column was passed: the
`logistic-multiclass` lane is not yet in `tools/identity_break.py`, so there
is no committed vendor JSON with its infer cell (the lane body is in
`docs/lanes/BRIEF_logistic_multiclass_2026-09-14.md`, section 5).

`check_apple-m4_sabotage.json`: verdict EXPECTED MISMATCH SEEN. Under the
descending dot-product fold of MOJOLEARN_HOST_SABOTAGE, `decision_function`,
`predict_proba` and the identity hash DIFFER on all 9 fixtures; `predict`
(the argmax label) is EQUAL on all 9, since a last-bit change of the logits
moves no argmax on these rows. The gate's verdict is on every surface
together, so the sabotage is caught.

NVIDIA and AMD recordings of this lane are owed, as for every classical
host lane; the CPU column has been checked on this M4 only.
