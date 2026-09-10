# SVC FAST on the Apple M4, 2026-09-10 (DEVIATIONS 2491, 2492)

`time_svc.py` is the program, `run1_m4.txt` its run: FAST Metal build of
main + these changes, scikit-learn 1.9.0 libsvm (single-threaded), HIGGS
prefixes with 28 features, RBF, C=1, gamma = 1/(d var), three alternating
pairs per shape after a warm-up, minimum per arm, accuracy on a 10,000-row
HIGGS tail beside every number.

| rows | sklearn fit | ours fit | sklearn predict 10k | ours predict 10k |
|---:|---:|---:|---:|---:|
| 5,000 | 606 ms | 144 ms | 1,716 ms | 20 ms |
| 20,000 | 10,561 ms | 531 ms | 6,561 ms | 68 ms |
| 50,000 | not run | 1,867 ms | not run | 121 ms |
| 100,000 | not run | 4,941 ms | not run | 194 ms |

Accuracy and support-vector counts agree with sklearn at both paired
shapes (0.6219/0.6218 and 0.6480/0.6481; 4242/4242 and 16220/16219 SVs).

Before these changes, same box, same session, not inside the paired
window: ours fit 1.66 s at 20,000 and 7.45 s at 50,000; predict 250 ms and
607 ms for 10,000 rows. The stage clock that located the two phases is
`MOJOLEARN_STAGE_TIMES=1` on the smo (see svm/README.md).

One GPU against one core: libsvm has no threading. The ratios are what
they are at these shapes and grow with rows; they are not a claim about
any other library or box.

## SVR, same session (`run1_svr_m4.txt`)

Same solver, same two changes. HIGGS label as the regression target,
epsilon 0.1, two alternating pairs, both pairs shown; RMSE on the same
10,000-row tail is equal to sklearn's at four decimals in every cell.

| rows | sklearn fit | ours fit | sklearn predict 10k | ours predict 10k | rmse both |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 695 / 701 ms | 295 / 211 ms | 1,780 / 1,788 ms | 21 / 98 ms | 0.5039 |
| 20,000 | 10,958 / 10,884 ms | 1,219 / 1,184 ms | 7,017 / 7,033 ms | 64 / 106 ms | 0.4867 |
