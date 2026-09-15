# HDBSCAN over the neighbors and hierarchy row drivers — two MI300X

RunPod pod `k0atgkbgayge5a`, two AMD Instinct MI300X (gfx942), 2026-09-14
22:56-22:59Z. Source: the `git archive` of commit `55ec5f108`
(`commit.txt`), which differs from the H100 leg's `ca6fb6473` only by that
leg's evidence directory; box and Mac source SHA256 agree. Body `body.sh`.

## Result (`out/gate.txt`)

- `tools/parallel_hdbscan_check.py`: `PASS 5 HDBSCAN configurations and 3
  refusals`; all trace records and attributes equal one device.
- Against the `-D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1` binding the check
  FAILS on its first two-device fit with the same MST refusal as on the H100s
  (`found 5 > 4`).
- `training/checks/graph_rows_check.mojo`: 6 PASS lines.
- `hdbscan/checks/hdbscan_check.mojo`: `hdbscan_check mode=IDENTICAL ALL OK`.

## Cross-vendor

`out/public.json` equals `../hdbscan-h100/out/public.json` as JSON: for all
five configurations the SHA256 over the 24 trace records and over the fitted
attributes is the same on two MI300X as on two H100s.

No speed or capacity claim.
