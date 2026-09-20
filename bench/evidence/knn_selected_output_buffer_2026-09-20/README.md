# KNN classifier selected output allocation (2026-09-20)

`KNeighborsClassifier._predict` formerly allocated both its complete Int32
label matrix and its complete Float32 probability matrix on every call.  The
native GPU, resident-GPU, and host ABIs already specify that exactly one is
selected by `want_proba`; the other pointer is unread and is documented as a
one-element sentinel.  The Python boundary now implements that contract.
There is no arithmetic, model, dispatch, or reduction-order change.

Apple M4 Metal, IDENTICAL build, 64 reference rows, four features, five
neighbors, and 1,000,000 fixed query rows.  The old arm restores only the
removed zero-filled allocation around the same native call.  Balanced order
four-run medians for `predict` were:

| seed | classes | exact SHA-256 | old s | new s | old/new |
|---:|---:|---|---:|---:|---:|
| 23 | 2 | yes | .627236 | .622786 | 1.007x |
| 29 | 32 | yes | .667386 | .643167 | 1.038x |
| 31 | 8 | yes | .637939 | .638464 | .999x |

The 32-class isolated-process case reduced peak RSS from 450,396,160 to
322,912,256 bytes (127,483,904 bytes) and retained output digest
`f507f44997fb04c4579140f05c036faf4f8be9340c4e254b89eb99e5ab5d5370`.
For an eight-output, four-class `predict_proba`, eliminating the unused
8,000,000-element labels matrix reduced observed peak RSS from 932,134,912 to
914,718,720 bytes; output digest was exactly
`62f3a9cc10ea1e21acde8fc633c4501a0a9eba03b288f78f07f82bf98e5bc067`.
The probability result dominates RSS in that case, so the measured peak
understates the deterministic 32,000,000-byte allocation removal.

The focused test records allocation shapes in both public arms, verifies the
one-element sentinel, covers the query-count reconstruction in
`predict_proba`, and checks exact results before and after instrumentation.
