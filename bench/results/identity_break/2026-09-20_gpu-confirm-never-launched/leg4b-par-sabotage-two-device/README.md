# Four `par-*` device-axis sabotage arms, watched firing on two real devices

Pod `qclqc51enh4u07`, 2x RTX 4090 sm_89, 2026-09-20. DELETEd and verified gone
(204, then 404, then absent from the listing). All four columns
`complete: true`, `--repeats 2`, fixtures `base,ties`.

## The four-cell control

Every arm here is `if rank > 0, read one row early`, so it is **inert on one
device by construction**. That makes the one-device sabotage column the
control for the control: a move there would mean the define reached something
other than the shard loop, and the two-device result would be about that
instead.

| | one device | two devices |
|---|---|---|
| **clean** | hashes | hashes, IDENTICAL to one-device |
| **sabotage** | *same hashes as clean* | **the arm fires** |

`sab_diff_one.log` is the inert half: `IDENTICAL=10, REFUSED=2`, byte for byte
the same summary as `clean_diff.log`. Nothing moved on one device.

`so_sha256.clean.txt` against `so_sha256.sabotage.txt` differ on 8 binding
files, so the defines did reach the compiler.

## What fired, per lane

| lane | define | two-device sabotage result |
|---|---|---|
| `par-gmm` | `MOJOLEARN_GMM_PARALLEL_SABOTAGE` | `ValueError: fit_gaussian_mixture covariances_ and plain covariances_ differ: 232 bytes of 256` (base), `242 of 256` (ties) |
| `par-resample` | `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` | `ValueError: parallel bootstrap distribution and bootstrap distribution differ: 3564 bytes of 8192` (base), `2925 of 8192` (ties) |
| `par-hdbscan` | `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` | `Exception: mst: Number of edges found by MST is invalid ... (found 6000 > 5999)` |
| `par-kernel-ridge` | `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` | `Exception: kernel_ridge_fit_host: the ridged kernel matrix K + alpha I is NOT positive definite (info=317 ...)` |
| `par-graph-agglomerative` | `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` | **IDENTICAL x2** — `6c9e7daffb48878f` both sides. The arm did not move it. |
| `par-cholesky` | `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` | **no information** — refused in ALL FOUR columns on `_mojolearn_gp.so, which is not built`. The targeted build list in `wrapper.sh` omits `build_gp`; that is this leg's error, not a finding. |

`par-gmm` and `par-resample` were caught by the lanes' OWN `_same_bytes`
oracle — the sharded result stopped matching the plain fit, which is precisely
the proposition the lane exists to assert. `par-hdbscan` and
`par-kernel-ridge` were caught downstream by the implementation's own guards,
the perturbed shard having corrupted the MST edge count and the Cholesky
factorization. None of those four errors appears in the clean two-device
column or in either one-device column.

`MOJOLEARN_GMM_PARALLEL_SABOTAGE` and `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE`
had, per `docs/lanes/SABOTAGE_AUDIT_2026-09-16.md`, only ever been run as
`mojo run -D ...` against standalone check programs and had **never been
compiled into a binding**. This is the first run of either against the lanes.

## AND THE CENSUS WILL NOT COUNT ANY OF IT

`verification_matrix.negative_control_moves` credits a move only when the
sabotage cell's verdict is `MOVED`, `DIVERGENT`, `RELOAD-MOVED`,
`BATCH_MOVED` or `RLPAIR_MOVED`. All four of these read **REFUSED**, because
the lane raised rather than returning moved bytes. The rule is right — "a
refusal is a build that did not run, not arithmetic that changed" — and here
it excludes four arms that demonstrably did fire.

**That is structural, not bad luck.** A `par-*` lane's body asserts
`_same_bytes(sharded, plain)` before it hashes anything. An arm that moves the
sharded result makes that assertion fail, so the cell can only ever be
REFUSED. For this family of lanes, an arm that works and an arm that is broken
produce the same census verdict, and the only way to tell them apart is to
read the refusal text — which is what this directory is for.

So: **four arms watched firing; zero of them creditable as `seen(build)`.**
Anyone reconciling this record against the census should expect that and not
go looking for the missing credit.

## What is still not known

* Whether `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE` reaches
  `par-graph-agglomerative` at all. It read IDENTICAL, which is either
  reached-but-inert or never-reached, and this run does not distinguish them.
* Anything about `par-cholesky` — see the missing `build_gp` above.
* The other 37 of the 43 unwatched arms. They have no define of any kind;
  nothing here changes that.
