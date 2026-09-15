# Kernel methods at d24e39765 (two-context drains) — two MI300X, Cholesky gate FAILED again

Pod in `pod_id.txt`. The public kernel-methods check passed again (32
configurations, public.json equal to the H100 report) and the Cholesky
native gate failed again at `n513-r2` with the same two forward hashes, in
both of its runs. `chol_trace_digests.txt` equals the two-H100 digests of
`../kernel-methods-h100-drain/` on every line except `n513-r2-solve-many`,
the MI300X two-device solve. See `../cholesky-mi300x-diag/` for the sweep,
the diagnostics and the host-staged fix.
