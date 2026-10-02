# Apple FAST handoff (2026-10-02, cloud code session paused on the weekly API limit)

For the next code session. Read `CLAUDE.md`, `docs/apple-fast/NEXT_PASS.md` (origin/main), then this file. The rules are
unchanged: code only, never build/test/time, base on origin/main, every change FAST + Apple only behind a default-off
`-D MOJOLEARN_*` define, GPU only, never push to `main` or `lane/apple-fast`, never `--no-verify`. Per-lane briefs and the
shared brief are in `docs/apple-fast/briefs-2026-10-02/` on this branch (`claude/charming-volta-y568xd`); copy that directory
to `~/mojolearn-evidence/briefs-2026-10-02/` before relaunching lane subagents (they read it from there; `~` is `/root`
on the cloud box).

Main was never pushed to. Every lane below had `origin/main` (eabeff395) merged INTO it; the M3 manager merges winners.

## State of every lane branch (all `lane/apple-fast-<family>`, all merge cleanly with origin/main unless noted)

| family | head | state | what is left |
|---|---|---|---|
| resample | 50b96e795 | done | wait for M3 (7 taxi lines) |
| linear | 44ec8018d | done | wait for M3 (istella lines) |
| tsa | 21e28b348 | done: ARIMA Kalman WIP finished (LLONLY, EVAL_WS) | wait for M3 (3 autoarima taxi-hourly lines) |
| bayes | 0f7218b29 | done: NaN guard on main's grid kernels (BAYES_GRID_GUARD) | wait for M3 (bayes-br-guard-istella) |
| core | 67258a0df | done: glm WIP finished (OLS_FAST_DEVICE_CENTER) | wait for M3 (core-ols-dcenter-taxi) |
| trees-scan | 43430ca0f | done; its scan variant dropped (overlap settled) | M3 queue still lists old `MOJOLEARN_SCAN_U32_BLOCK` lines (now no-ops); manager re-reads trees-scan.txt |
| trees-depthwise | 92e87fe17 | done; keeps `MOJOLEARN_GBDT_CTR_FAST_SCAN` | wait for M3 (3 gbdt-categorical taxi lines) |
| trees-ensembles | d7887faa1 | done; env switches -> define bit set; BAG_SESSION dropped (main covers it) | wait for M3 (8 lines) |
| trees-io | 9d916e91f | done | wait for M3 (5 lines) |
| rfet-scan | b0d5b1fab | docs merge only | nothing |
| trees-symmetric | ae0774468 | done: SYM_DEVICE_LEVEL two-stage level score | wait for M3 (tsym-level-istella) |
| yetirank | c9eb1ce14 | done; also carries trees-yeti; SCORE_GRID + SYM_HIST_UNROLL8 | wait for M3 (2 istellarank lines) |
| trees-yeti | b157e5ee0 | merge only | nothing (yetirank builds on it) |
| prep | 387211293 | done (PREP_FAST_MINMAX) | wait for M3 (5 lines) |
| prep2 | cc3b27d5f | done (PREP2_FAST_EIGH_BLOCK) | wait for M3 (7 lines) |
| gram | 033f6c096 | done (3 defines) | wait for M3 (6 lines) |
| kernel | 4e08139d4 | done; carries gram; one shared grid Gram | wait for M3 (4 istella lines) |
| kapprox | 015510968 | done (KAPPROX_DEVICE, chi2 samplers) | after a win: second datasets. CHECK overlap with neighbors2/isotonic-knn: it regenerated the x_neighbors bindings via x_neighbors/gen.py |
| select | cb5c31a60 | done (SELECT_FREG/FCLS/D); prep2 merged in to settle x_prep/device.mojo | a stray `merge.log` is committed at the branch root: delete it in the next commit |
| cluster | 2d5b3dffa | done (2 defines) | TRIM docs/apple-fast/ab/cluster.txt: one dataset per define, drop the switch-less `cluster-cc/aggl/spec-fast-*` baseline lines |
| cluster2 | 670baf47a | done; carries cluster; 5 defines; OPTICS one-block order rewritten as per-step grid | TRIM cluster2.txt the same way; drop `cluster2-bgmm-phases-istella` (single arm) |
| meta | 497bcad7d | calibrated done (MOJOLEARN_CALIB_GNB_FOLDS, x_prep fold program), request line + afc_ab_def.sh | multioutput-reg (taxi 2.1x) NOT started: see briefs/meta.md item 2 |
| decomp-linalg | 8fdbdd0e8 | main merged and pushed | write docs/apple-fast/ab/decomp-linalg.txt (light form, incl. the tiled lstsq arm on istella) and decomp-linalg.md; the WIP commit fab0562ed's partial .txt needs checking |
| decomp-sparse | e021e3498 | main merged and pushed | check its ab file is in the light form; `git merge-tree --write-tree` it against decomp-linalg and settle if it conflicts |
| ann | 59afb759a | main merged, env reads removed (IVF_FAST_DEVICE_* defines), pushed | check ann.txt is in the light form (afc_ab_def.sh for define switches) |
| isotonic-knn | 4f4cd51e2 | main merged and pushed | check its ab file; merge-tree against neighbors2 and kapprox |
| depthwise | 341ab92ea | WIP: `gbdt/methods/greedy_subsets_searcher/kernel/dw_tree_sync.mojo` committed, NOT referenced | finish item 7 (one wait per tree, `-D MOJOLEARN_GBDT_DW_TREE_SYNC`), hook it into the fused chain, add the aft_ab line (see briefs/depthwise.md) |
| dart | c7fe11e34 | WIP: `xtrees/dart_device.mojo`, x_trees binding and `_expansion_trees.py` hook started | finish per briefs/dart.md; add dart.txt (aft_ab.sh, dart and dart-reg on istella) |
| ets | 7433af169 | WIP: `sequence/ets_team.mojo` committed, NOT referenced | finish per briefs/ets.md (damped-ets candidates x series on one grid); add ets.txt |
| neighbors2 | origin 67157e40a (NOT merged with main) | the main merge exists only as a local commit 6aff39c27 and as the patch `briefs-2026-10-02/neighbors2-unpushed-merge.patch` here | the pre-push hook refuses it: 13 host-route findings, `_buf`/`_down` host calls at x_neighbors/iter_device.mojo:1869-2005 and a one-block launch at x_neighbors/svgp_fast.mojo:110. Redo the merge (or apply the patch on top of a fresh `git merge origin/main`), replace those calls with main's device-side idiom (direct `ctx.enqueue_create_buffer` / `enqueue_copy`, grid launches), convert the SVGP env switch to a define, then push normally |

## Overlaps already settled
trees-scan vs trees-depthwise (kept depthwise's `GBDT_CTR_FAST_SCAN`); gram -> kernel; cluster -> cluster2; prep2 -> select;
trees-yeti -> yetirank. Verified mutually clean: gram/kernel/bayes/linear/core (x_linear), prep x prep2, cluster x cluster2,
trees-scan x trees-depthwise. Still to verify: kapprox x neighbors2 x isotonic-knn (x_neighbors bindings), decomp-linalg x
decomp-sparse (x_decomp/device.mojo).

## After the M3 builds
Read `origin/lane/apple-fast-results:docs/apple-fast/m3/build-errors.txt` first (grep, never cat), then `results.txt`.
Every lane's `docs/apple-fast/ab/<family>.md` lists its risky compile sites. Known risk pattern: bare `buf.unsafe_ptr()`
passed to a helper typed `MutPointer[T, MutAnyOrigin]` (use `.unsafe_origin_cast[MutAnyOrigin]()`); the tier branch still
fails on this at core/gemm.mojo:411.

## Housekeeping
Commit trailers: the session's attribution lines (`Co-Authored-By` + `Claude-Session`) are required by the harness; the
"no model name" rule in the shared brief was over-strict and some lanes dropped the trailer. Either form is acceptable.
