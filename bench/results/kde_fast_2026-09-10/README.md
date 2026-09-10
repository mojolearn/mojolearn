# KernelDensity FAST score pass (DEVIATION 2490), Apple M4, 2026-09-10

`time_kde.py` is the program; `run1_m4.txt` its run (FAST Metal build of
main + this change, scikit-learn 1.9.0 KDTree at `rtol=atol=0`, one core,
five alternating pairs after one warm-up per arm, MINIMUM per arm).

Headline, 16,000 x 16,000 x 8: scikit-learn 7,671 ms, ours 30.3 ms.
Before this change the same call on the same box took 540.7 ms (staged
path, measured in the same session but not inside the paired window), so
the fused pass is about 17.8x less time for our own arm and the sklearn
ratio moved from about 14x to about 250x. 100,000 x 100,000 x 8 scores in
499 ms; the staged path could not allocate its two 40 GB matrices there.

Scope: the sklearn arm is single-threaded by construction (KDTree has no
`n_jobs`), so this is one GPU against one core. Wide rows (d > 64) take the
chunked kernel at about 0.6 Gcells/s at d=100; that arm is correct and no
longer catastrophic but is not tuned. IDENTICAL and DETERMINISTIC are
unchanged (13 of 13 `kde/checks/kde_check.mojo` checks OK under
`-D MOJOLEARN_NUMERIC_IDENTICAL=1`, bit-equal to the oracle).

Correctness evidence for the FAST arm: fused vs staged over 360 cases
(6 metrics x 6 kernels x weighted/unweighted x 5 shapes with d in
{1,3,5,33,64}) agree on every sentinel cell and within 7e-5 relative on
every finite cell; sklearn agreement within 3.2e-6 on the finite cells;
`kde/checks/kde_check.mojo` 13 of 13 OK in FAST (one `mojo run` of it
crashed once in the Metal driver at process exit and passed on the next
two runs and as a built binary).
