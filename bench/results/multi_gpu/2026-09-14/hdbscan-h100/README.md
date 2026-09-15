# HDBSCAN over the neighbors and hierarchy row drivers — two H100s

RunPod pod `aaecwt58ggmrz4`, two NVIDIA H100 80GB HBM3 (sm_90a), 2026-09-14
22:35-22:37Z. Source: the `git archive` of commit `ca6fb6473`
(`commit.txt`); box and Mac source SHA256 agree. Body `body.sh`; GPU work
serial. Design: `docs/multi_gpu/hdbscan.md`.

## Result (`out/gate.txt`)

- `tools/parallel_hdbscan_check.py`: `PASS 5 HDBSCAN configurations and 3
  refusals` (`out/public.json`). For 5, 37, 200, 515 and 1024 rows (eom and
  leaf selection, alpha 1.5, allow_single_cluster, a duplicated point), all
  24 identity-trace records of the one-device fit equal those of the
  two-device `fit_hdbscan` fit, and labels, core distances, cluster, outlier,
  Boruvka-round and condensed-cluster counts are equal.
- Against an HDBSCAN binding built with `-D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1`
  (later distance-row owners read their rows one row early) the check FAILS
  on its first two-device fit: the corrupted distance rows make the MST
  refuse, `mst: Number of edges found by MST is invalid ... (found 5 > 4)`,
  where the one-device fit succeeded. This shows the dense distance rows run
  through the hierarchy row driver. No separate reach witness was built for
  the k-NN rows.
- `training/checks/graph_rows_check.mojo`: 6 PASS lines (raw distance and
  selection bits of the neighbors and hierarchy row drivers).
- `hdbscan/checks/hdbscan_check.mojo`, the unchanged single-device gate:
  `hdbscan_check mode=IDENTICAL ALL OK`.

No speed or capacity claim; the root holds the m x m graph.
