# Apple FAST experiments: every A/B, kept and dropped

The record of every Apple FAST (M3 Ultra) experiment from Oct 2 to Oct 3, 2026: the winners that became defaults, the losers, and the A/Bs still owed.
Sources: `~/mojolearn-evidence/apple-fast/LEDGER.md` (every KEEP / DROP / MERGED / CANDIDATE line) and the queue lines (`docs/apple-fast/ab/*.txt`) and notes (`docs/apple-fast/notes/*.md`) on each lane branch.

Process: [experiment validation, toggle lifecycle and promotion](EXPERIMENT_PROCESS.md).

## How to use it

- **Before you write a new experiment, search this file for the define** (and for the algorithm). If it was dropped, read the reason first. Do not re-run a dropped idea unless the code it touched has changed since.
- **To get dropped code back**, use the branch and sha in its row: `git show <sha>:<file>`, or `git diff origin/main...<sha> -- <dir>` for the whole change. The lane branches stay on origin.
- **Before/after** is milliseconds on the M3 Ultra, FAST mode. Arm A is FAST main code; arm B adds the define. n=1 or n=2 per arm, as the ledger notes. Board numbers are in `BOARD_M3_FAST.md`.
- **Lesson (Oct 3):** an A/B on a branch whose base is older than main can show a gain main already has. Judge such a gain against main's board, or re-run the A/B with main merged in.

## Policy

- **Winners** become FAST + Apple defaults. Each one gets a `<NAME>_OFF` define that restores the old path, and a code comment citing its A/B (tag and numbers). IDENTICAL mode never changes.
- **Losers are not kept in main's code.** They stay on their branch at the recorded sha. A dropped define that is still on main is a dead toggle; a cleanup lane removes those.
- **User clarification (Oct 4):** "it is OK if bits change for fast work.. do you understand that? but quality cannot go down beyond noise". FAST does not require bit identity; judge quality using the relevant metric and measured noise. A quality change beyond noise is DROPPED-quality; a bit change alone is not a failure. A slower arm with better quality can be KEPT (see `X_PREP_CLASS_COV_GRID`).
- Keep concise evidence comments next to surviving failed or held opt-in toggles: dataset, A/B tag, measured effect, status, and this record. Distinguish an inconclusive/noise result or an old-base hold from an established regression; a bundle failure does not prove each component failed alone. Do not restore deleted code to annotate it.

## Verdicts

| verdict | meaning | rows |
|---|---|---|
| KEPT `<main sha>` | FAST + Apple default since that main commit | 84 |
| DROPPED-slower | B slower than A | 19 |
| DROPPED-noise | the difference is inside run-to-run spread (arms overlap, or under 5% at n=1), or the signs are mixed across datasets | 48 |
| DROPPED-quality | faster, but the quality metric got worse | 3 |
| DROPPED-semantics | no longer matches main's code path: stale base, duplicate of a main change, a no-op, or a host step in the GPU path | 4 |
| MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | merged to main default OFF without an A/B (Andrew Oct 3: "just merge in the neurals provided they compile; not publicly exposed; don't lose the work") | 84 |
| OPEN | not measured yet, measured but not judged, or the run did not finish | 189 |

Each row is one define, or one combination of defines, on one branch. Combination rows (`A + B`) are B arms that turn on several defines together. `(_OFF)` in a define name means the switch on main is the opt-out. Shas are `git rev-parse --short origin/<branch>` as of this record; the ledger's measured head is given where it differs.

## Trees (101)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `TREESHAP_FAST_TABLE` | TreeExplainer (tree-shap) / taxi, istella | lane/apple-fast-fix-treeshap @ 8fcd690cb | ab-tshap-table-taxi, ab-tshap-table-istella | taxi 21.4 -> 11.8; istella 71.6 -> 25.1 | KEEP (FAST+Apple default, `_OFF`) | additivity error identical; per-(leaf, one-fraction pattern) term table built once per call; each (row, tree) walks each leaf's path for its pattern and adds n stored terms, cells in registers (no per-row extend/unwound, cubic in path length); same terms in the same order = same bits; expect several-fold on depth-6 forests |
| (no switch) YetiRank 256-thread block kernel | yetirank / istellarank | lane/apple-fast @ 46f0cf09f | aft-ab-yeti1 | 55,385 -> 8,290 | KEPT 269ffa57a | 6.7x; same hash |
| `DART_DEVICE` | dart, dart-reg / istella | lane/apple-fast-dart @ b192c3353 | dart-dev-istella, dartreg-dev-istella | dart istella 45,837 -> 24,730; dart-reg istella 45,445 -> 24,837 | KEPT 741a5495d | -46% / -45%; acc .9487 -> .9486 |
| `ET_PART_ROWS (_OFF)` | et / taxi, istella | lane/apple-fast @ 269ffa57a | aft-ab-etpr | taxi 3,866 -> 3,036; istella 4,369 -> 4,140 | KEPT 269ffa57a | -21% / -5%; same hashes |
| `FOREST_DEVICE_FINITE (_OFF)` | et, rf / taxi, istella | lane/apple-fast-rfet-scan @ 500168cfe | aft-ab-etfin, aft-ab-rffin | et taxi 2,867 -> 2,829, istella 4,143 -> 4,059; rf 10,243 -> 10,087 | KEPT 69f7a41fd | et -1.3% / -2.0%; rf noise; refusals pass |
| `GBDT_CTR_FAST_FREQ` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-freq | categorical 33,042 -> 27,761 | KEPT 0ad05a65f | -16.0% (merged from lane/apple-fast-ctrfreq) |
| `GBDT_CTR_PERM_BATCH` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-permbatch | categorical taxicat 36,772 -> 35,780 | KEPT 57ddea946 | -2.7%; both B below both A |
| `GBDT_DW2_PART_VEC4` | depthwise / istella, taxi | lane/apple-fast-dwgap2 @ 23eca012b | dw2-part-vec4-taxi, dw2-part-vec4-istella | depthwise taxi 12,711 -> 11,361; istella 17,154 -> 16,539 | KEPT c55b8c377 | -10.6% / -3.6% |
| `GBDT_DW2_SCAN_SMEM` | depthwise / istella, taxi | lane/apple-fast-dwgap2 @ 23eca012b | dw2-scan-smem-taxi, dw2-scan-smem-istella | depthwise taxi 13,861 -> 13,127; istella 17,237 -> 17,014 | KEPT c55b8c377 | -5.3% / -1.3% |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC` | depthwise / taxi, istella | lane/apple-fast-depthwise @ 4547e0d14 | dw-fcns-taxi, dw-fcns-istella | taxi 14,933 -> 14,120; istella 19,561 -> 19,522 | KEPT 745ff31da | -5.4% / neutral; depthwise only |
| `GBDT_DW_MODE_SKIP` | depthwise / istella, taxi | lane/apple-fast-dwgap @ 0256457c3 | dwgap-modeskip-taxi, dwgap-modeskip-istella | depthwise taxi 13,744 -> 12,930; istella 17,297 -> 16,975 | KEPT 7736fa5b8 | -5.9% / -1.9%, bit-identical by construction; also reaches lossguide |
| `GBDT_LG_EXACT_BATCH128` | lossguide / taxi | lane/apple-fast-lgw128 @ ec921c123 | aft-ab-lgw128 | taxi 13,148 -> 12,921 | KEPT 9722e5a2b | -1.7%, arms separate; auc same |
| `GBDT_LG_EXACT_BATCH64` | lossguide / taxi, istella | lane/apple-fast-trees2 @ bfd1d7cc6 | aft-ab-lgw64 | taxi 16,776 -> 14,900; istella 22,886 -> 20,715 | KEPT 269ffa57a | -11% / -9.5%; superseded by EXACT_BATCH128 (9722e5a2b) |
| `IF_ROWMAJOR (_OFF) + device finite scan` | iforest / istella, taxi | lane/apple-fast-trees2 @ bfd1d7cc6 | aft-ab-if2 | istella 264.6 -> 261.9; taxi 86.6 -> 85.4 | KEPT 269ffa57a | 1%, small; NaN/inf refusal still raises |
| `MULTICLASS_HESSIAN_BATCH` | pairlogit / multiclass / taxi | lane/apple-fast-pairlogit @ 00bfe78f8 | mc-hessbatch-taxi | multiclass taximc 18,603 -> 17,857 | KEPT 28adffe85 | -4.0%; mlogloss identical |
| `ORDERED_BATCH_EST` | ordered / taxi | lane/apple-fast-ordered @ 5b8722353 | ord-be-taxi | ordered taxi 275,854 -> 264,146 | KEPT cf45a2732 | auc .6289 -> .6285; both B runs below both A |
| `PAIRLOGIT_EST_REUSE + PAIRLOGIT_GROUP_FUSED` | pairlogit / multiclass | lane/apple-fast-pairlogit @ 00bfe78f8 | pl-estreuse | 3,765 -> 3,480 (on GROUP_FUSED) | KEPT 28adffe85 | -7.6%; same hash |
| `PAIRLOGIT_GROUP_FUSED` | pairlogit / multiclass | lane/apple-fast-pairlogit @ 00bfe78f8 | pl-groupfused | pairlogit istellarank 4,802 -> 3,760 | KEPT 28adffe85 | -21.7%; ndcg10 identical |
| `RF_HIST_COLUMNS8 (_OFF)` | rf / taxi, istella | lane/apple-fast @ 269ffa57a | aft-ab-rfc8 | taxi 11,513 -> 11,361; istella 14,377 -> 14,235 | KEPT 269ffa57a | -1.3% / -1.0%, arms do not overlap |
| `SEG_SUMS_BLOCK_SCAN (_OFF)` | rf / taxi, istella | lane/apple-fast-rfet-scan @ 500168cfe | aft-ab-rfseg | istella 13,583 -> 10,116; taxi 10,741 -> 10,719 | KEPT 69f7a41fd | -25.5% istella; same hash; also resample |
| `TE_ADA_SESSION` | adaboost-clf / taxi; adaboost-reg / taxi | lane/apple-fast-trees-ensembles @ 1669a3bcd | te-adaclf-taxi, te-adareg-taxi | adaboost-clf taxi 2,279 -> 2,073; adaboost-reg taxi 3,355 -> 1,959 | KEPT ed49f2b11 | -9% / -42%; quality identical |
| `TE_ADA_SESSION + TE_ADA_SESSION_SHARE` | adaboost-clf / taxi | lane/apple-fast-trees-ensembles @ 1669a3bcd | te-adaclf-share-taxi | adaboost-clf taxi 2,056 -> 1,608 (on SESSION) | KEPT ed49f2b11 | -22%; acc identical |
| `TE_NATIVE_SPLITS` | calibrated / istella; multioutput-clf / istella; ovr / istella; stacking-clf / istella;... | lane/apple-fast-trees-ensembles @ 1669a3bcd | te-stackclf-istella, te-stackreg-istella, te-cal-istella (+2) | stacking-clf istella 9,875 -> 9,370; calibrated 4,992 -> 4,515; ovr 5,380 -> 5,259; multioutput-clf 2,219 -> 2,088; stacking-reg 10,617 -> 10,309 | KEPT ed49f2b11 | -2% to -10%; quality identical |
| `YETI_EST_REUSE_SEARCH (_OFF)` | yetirank / istellarank | lane/apple-fast @ 098e89988 | aft-ab-yreuse1 | 8,271 -> 5,598 | KEPT 269ffa57a | -32%; ndcg10 .68081 -> .68120 |
| `YETI_FAST_SORT` | yetirank / istellarank | lane/apple-fast-yetirank @ c7b35fd7c | aft-ab-ysort1 | 5,303 -> 3,350 | KEPT 79691ec6e | -36.8%; ndcg identical |
| `YETI_SEARCH_TASK16K` | yetirank / symmetric | lane/apple-fast-trees-yeti @ 65f551e39 | yeti-task16k | yetirank istellarank 5,315 -> 5,102 | KEPT eb3fca1ac | -4.0%; yields to YETI_FAST_SORT (16K kernel lacks the sort) |
| `ET_DEVICE_BATCH_65536` | et / taxi | lane/apple-fast @ 269ffa57a | aft-ab-etb64, aft-ab-etb64b | on PART_ROWS: taxi 3,041 -> 3,080 | DROPPED-slower | +1.3%; code removed from main e242fb001; recover at lane/apple-fast@269ffa57a |
| `ET_TPB_256` | et / taxi, istella | lane/apple-fast @ 269ffa57a | aft-ab-ettpb, aft-ab-ettpb2 | on PART_ROWS: taxi 3,030 -> 3,072; istella 4,133 -> 4,379 | DROPPED-slower | +1.4% / +6% (alone it was mixed); code removed from main 4a78e109a; recover at lane/apple-fast@269ffa57a |
| `GBDT_CTR_FAST_FREQ + GBDT_CTR_FAST_SCAN` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-both | categorical 32,739 -> 27,662 | DROPPED-noise | same as FREQ alone; SCAN adds nothing; code removed from main afda7b9ce; recover at lane/apple-fast-trees-depthwise@f743edd60 |
| `GBDT_CTR_FAST_SCAN` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-scan | categorical 32,892 -> 33,044 | DROPPED-noise | +0.5%, overlap; code removed from main afda7b9ce; recover at lane/apple-fast-trees-depthwise@f743edd60 |
| `GBDT_CTR_PERM_PTRS` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-permptrs | categorical taxicat 37,098 -> 36,618 | DROPPED-noise | -1.3%, B runs straddle A; code removed from main 4722f0a88; recover at lane/apple-fast-trees-depthwise@f743edd60 |
| `GBDT_DW2_COPY_ZERO` | depthwise / istella, taxi | lane/apple-fast-dwgap2 @ 23eca012b | dw2-copy-zero-taxi, dw2-copy-zero-istella | depthwise taxi 13,051 -> 13,441 | DROPPED-slower | +3.0%, arms overlap; code removed from main 4037c6b9a; recover at lane/apple-fast-dwgap2@23eca012b |
| `GBDT_DW_FAST_DEV_SCALE` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-devscale-dwtaxi | 10,812 -> 10,780 | DROPPED-noise | -0.3%, overlap |
| `GBDT_DW_FAST_SKIP_FINAL_STATS` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-skipfs-dwtaxi | 11,108 -> 11,048 | DROPPED-noise | -0.5% |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC` | depthwise / taxi | lane/apple-fast-depthwise @ 4547e0d14 | dw-tree-taxi | depthwise taxi 14,453 -> 14,386 | DROPPED-noise | -0.5%, B runs straddle A; code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14 |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC + GBDT_DW_TREE_SYNC_CHECK` | depthwise / taxi | lane/apple-fast-depthwise @ 4547e0d14 | dw-tree-check-taxi | depthwise taxi 14,485 -> 14,581 | DROPPED-noise | +0.7%; GBDT_DW_TREE_SYNC: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14; GBDT_DW_TREE_SYNC_CHECK: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14 |
| `GBDT_LG_EXACT_BATCH16` | lossguide / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-lgw16 | taxi 16,776 -> 18,093 | DROPPED-slower | +8%; code removed from main dc23f13c8; recover at lane/apple-fast-trees2@50dfdcca0 |
| `GBDT_QH_FAST_FUSED_Q` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-fusedq-dwtaxi | 10,673 -> 10,783 | DROPPED-noise | +1%, overlap |
| `GBDT_SEG_SUMS_BLOCK` | depthwise / taxi | lane/apple-fast-rfet-scan @ 500168cfe | aft-ab-gbseg | 10,325 -> 10,293 | DROPPED-noise | -0.3%, overlap; stays opt-in; code removed from main 7ca37fd0b; recover at lane/apple-fast-rfet-scan@500168cfe |
| `GBDT_SM_X4` | depthwise / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-smx4 | taxi 14,881 -> 14,772 | DROPPED-noise | arms overlap; code removed from main e1b520e88; recover at lane/apple-fast-trees2@50dfdcca0 |
| `GBDT_SM_X8` | lossguide / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-smx8 | taxi 16,483 -> 19,577 | DROPPED-slower | +19%; code removed from main eccc9be7e; recover at lane/apple-fast-trees2@50dfdcca0 |
| `IF_QUERY_RAW` | iforest | lane/apple-fast-trees-io @ df6a77c21 | trees-io-ifq-build | iforest taxi 328 -> 324; score istella 2,290 -> 2,289 | DROPPED-noise | A runs straddle; refusals ok |
| `IF_SAMPLED_UPLOAD` | iforest / istella | lane/apple-fast-trees-io @ df6a77c21 | trees-io-if-istella | iforest istella 384 -> 385 | DROPPED-noise | same hash |
| `ORDERED_FOLD_DERIVS` | ordered / taxi | lane/apple-fast-ordered @ 5b8722353 | ord-fd-taxi | ordered taxi 282,245 -> 287,217 | DROPPED-slower | +1.8%; code removed from main 219ae194a; recover at lane/apple-fast-ordered@5b8722353 |
| `REORDER_FLAGS_SCAN_BLOCK + SEG_SCAN_BLOCK` | depthwise / rf / taxi | lane/apple-fast-trees-scan @ 43430ca0f | trees-scan-dw-taxi | depthwise taxi 12,631 -> 12,853 | DROPPED-noise | +1.8%, overlap |
| `RF_NODESPLIT_ZERO_AFTER_READ + RF_FAST_BATCH16K` | rf / taxi, istella | lane/apple-fast-trees2 @ bfd1d7cc6 | aft-ab-rf1 | taxi 11,502 -> 11,485; istella 14,324 -> 14,312 | DROPPED-noise | 0.1%; RF_FAST_BATCH16K: code removed from main 288809ef8; recover at lane/apple-fast-trees2@bfd1d7cc6; RF_NODESPLIT_ZERO_AFTER_READ: code removed from main 931143bf8; recover at lane/apple-fast-trees2@bfd1d7cc6 |
| `RF_SMALL_NODE_1024` | rf / taxi, istella | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-rfsn | taxi 11,491 -> 11,482; istella 14,320 -> 14,329 | DROPPED-noise | ; code removed from main baa477878; recover at lane/apple-fast-trees2@50dfdcca0 |
| `SEG_SCAN_BLOCK` | depthwise / rf / istella | lane/apple-fast-trees-scan @ 43430ca0f | trees-scan-rf-istella | rf istella 13,704 -> 10,146 | DROPPED-semantics | stale base: main already had the gain via SEG_SUMS_BLOCK_SCAN (69f7a41fd); duplicate |
| `SYM_DEVICE_LEAVES` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-leaves-istella | symmetric istella 14,671 -> 14,643 | DROPPED-noise | -0.2%, overlap |
| `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-all-1000-istella | symmetric-1000 istella 32,932 -> 33,146 | DROPPED-noise | +0.6%; no switch in this lane wins |
| `SYM_DEVICE_LEVEL` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-level-istella | symmetric istella 16,926 -> 16,852 | DROPPED-noise | -0.4%; auc equal |
| `SYM_DEVICE_PARTITION` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-part-istella | symmetric istella 16,869 -> 17,081 | DROPPED-slower | +1.3% |
| `SYM_HIST_FAST` | yetirank / symmetric / istella, taxi | lane/apple-fast-trees-yeti @ 65f551e39 | yeti-symhist, yeti-symhist-sym-taxi, yeti-symhist-sym-istella, yeti-symhist-ordered-taxi | symmetric istella 14,710 -> 14,685; ordered taxi 66,705 -> 67,150 | DROPPED-noise | -0.2% / +0.7%; code removed from main 6f5ace7fa; recover at lane/apple-fast-trees-yeti@65f551e39 |
| `SYM_HIST_FAST + YETI_SEARCH_TASK16K` | yetirank / symmetric | lane/apple-fast-trees-yeti @ 65f551e39 | yeti-both | - | DROPPED-noise | SYM_HIST_FAST part dropped (see above); code removed from main 6f5ace7fa; recover at lane/apple-fast-trees-yeti@65f551e39 |
| `SYM_HIST_FAST + YETI_SYM_HIST_UNROLL8` | yetirank | lane/apple-fast-yetirank @ c7b35fd7c | yeti-h8unroll | yetirank 5,315 -> 5,373 | DROPPED-slower | +1.1%; SYM_HIST_FAST: code removed from main 6f5ace7fa; recover at lane/apple-fast-yetirank@c7b35fd7c; YETI_SYM_HIST_UNROLL8: code removed from main 4f634c5d0; recover at lane/apple-fast-yetirank@c7b35fd7c |
| `SYM_NO_TAIL_DRAIN` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-notail-istella | symmetric istella 16,998 -> 17,106 | DROPPED-noise | +0.6% |
| `YETI_TREE_SEARCH_SCORE_GRID` | yetirank | lane/apple-fast-yetirank @ c7b35fd7c | yeti-scoregrid | yetirank 5,304 -> 5,288 | DROPPED-noise | -0.3%, overlap; code removed from main 99da8a08f; recover at lane/apple-fast-yetirank@c7b35fd7c |
| `CTR_INDEX_FUSED` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-index-fused-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `CTR_ONEHOT_DEVICE` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-onehot-device-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `CTR_PREP_SHARED` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-prep-shared-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `CTR_SORT_ONCE` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-sort-once-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `EST_REUSE_PART` | symmetric / istella | lane/apple-fast-sym-est @ c8518eb52 | sym-est-rp-ist | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `EST_SHRINK_FUSED` | symmetric / istella | lane/apple-fast-sym-est @ c8518eb52 | sym-est-sh-ist, sym-est-sh-1k-ist | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `EST_STATS_FUSED` | symmetric / istella | lane/apple-fast-sym-est @ c8518eb52 | sym-est-sf-ist | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_BOOT_DEVICE` | symmetric / istella | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-boot-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_EVAL_FUSED` | symmetric / istella | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-evfused-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_EVAL_SKIP_EMPTY` | symmetric / istella, taxi | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-evskip-istella, sym-feat-evskip-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_INDEX_PACK_DEVICE` | symmetric / istella, taxi | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-pack-istella, sym-feat-pack-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_PREDICT_PACKED` | symmetric / istella, taxi | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-pred-istella, sym-feat-pred-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `GBDT_QUANT_DEVICE` | symmetric / istella, taxi | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-quant-istella, sym-feat-quant-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MC_CLASS_BATCH_DERIV` | multiclass / pairlogit / yetirank / istella, taxi | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-mc-deriv-istella, symmulti-mc-deriv-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MC_CLASS_BATCH_DERIV + MC_CLASS_BATCH_EST` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-mc-both-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MC_CLASS_BATCH_EST` | multiclass / pairlogit / yetirank / istella, taxi | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-mc-est-istella, symmulti-mc-est-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ORD_ALL` (`_OFF`) | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-all-taxi, sym-ordered-all-istella | istella 75,758 -> 63,554; taxi 56,473 -> 48,163 | KEEP gated (FAST+Apple default when n_features > 32, `-D MOJOLEARN_ORD_ALL_OFF`; lane/apple-fast-ordall) | istella -16.1%, auc .979518 -> .979529, logloss .190603 -> .190385 (better); taxi -14.7% but auc .628875 -> .628572, logloss .52909 -> .529215 (worse) = DROPPED-quality on narrow data, so runtime gate `len(layout.features) > ORD_ALL_MIN_FEATURES` (32) keeps taxi on main's path; the four pieces below are its internal parts, no standalone defines |
| `ORD_FOLD_BINS_ONE` | ordered / taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-fbo-taxi | taxi -1.0% | DROPPED-noise (standalone) | define removed; the code stays as a piece of `ORD_ALL` |
| `ORD_FOLD_INDEX` | ordered / taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-fidx-taxi | taxi -5.9% | DROPPED-quality (standalone) | auc down on taxi; define removed; the code stays as a piece of `ORD_ALL` (wide data only) |
| `ORD_STD_PARALLEL` | ordered / istella | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-std-istella | istella -1.8% | DROPPED-noise (standalone) | define removed; the code stays as a piece of `ORD_ALL` |
| `ORD_TREE_LEAN` | ordered / istella | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-lean-istella | istella ~-0.8% | DROPPED-noise (standalone) | define removed; the code stays as a piece of `ORD_ALL` |
| `PL_GROUP_NARROW` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-pl-narrow-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PL_GROUP_NARROW + PL_PAIRS_ONCE` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-pl-both-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PL_PAIRS_ONCE` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-pl-once-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SHAP_KERNEL_DEV` | kernel-shap / istella | lane/apple-fast-shap @ 13343dd51 | shap-kernel-dev | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SHAP_PERM_CACHE` | permutation-shap / istella | lane/apple-fast-shap @ 13343dd51 | shap-perm-cache | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SHAP_TREE_TAB` | tree-shap / istella | lane/apple-fast-shap @ 13343dd51 | shap-tree-tab | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_BUF_ARENA` | symmetric / taxi | lane/apple-fast-sym-iter @ 4956a2234 | sym-iter-arena-1000-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_CTR_ALL` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-all-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_CTR_PERM_BATCH` | categorical / taxicat | lane/apple-fast-sym-ctr @ 39c3c9daf | sym-ctr-perm-batch-taxicat | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_DERIV_FUSED` | symmetric / istella, taxi | lane/apple-fast-sym-iter @ 4956a2234 | sym-iter-fused-1000-taxi, sym-iter-fused-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_EST_ALL` | symmetric / istella, taxi | lane/apple-fast-sym-est @ c8518eb52 | sym-est-all-ist, sym-est-all-1k-ist, sym-est-all-1k-taxi, sym-est-all-ord-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_FEAT_ALL` | symmetric / istella, taxi | lane/apple-fast-sym-feat @ bca0e3a48 | sym-feat-all-istella, sym-feat-all-taxi, sym-feat-all-1000-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_GATHER_FUSED` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-gather-fused-istella, symhist-gather-fused-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_HIST_ALL` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-all-istella, symhist-all-taxi, symhist-all-1000-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_HIST_MULT` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-hist-mult-istella, symhist-hist-mult-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_ITER_ALL` | symmetric / istella, taxi | lane/apple-fast-sym-iter @ 4956a2234 | sym-iter-all-1000-taxi, sym-iter-all-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_LEAF_FROM_STATS` | symmetric / istella, taxi | lane/apple-fast-sym-iter @ 4956a2234 | sym-iter-leaf-1000-taxi, sym-iter-leaf-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_MULTI_ALL` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-all-mc-istella, symmulti-all-pl-istella, symmulti-all-yr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_PART_STATS_PAR` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-part-stats-par-istella, symhist-part-stats-par-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_RESOLVE_BLOCK` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-resolve-block-istella, symhist-resolve-block-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_REUSE_PARTITION` | symmetric / taxi | lane/apple-fast-sym-iter @ 4956a2234 | sym-iter-reuse-1000-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_SCAN_SUB_FUSED` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-scan-sub-fused-istella, symhist-scan-sub-fused-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SYM_SORT_SWAP` | symmetric / istella, taxi | lane/apple-fast-sym-hist @ 3bb4db314 | symhist-sort-swap-istella, symhist-sort-swap-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `YR_TASK_FUSED` | multiclass / pairlogit / yetirank / istella | lane/apple-fast-sym-multi @ d2c832da0 | symmulti-yr-fused-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |

## Linear (46)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `SGD_FAST_PS_SIMD` | sgd-ocsvm / taxi, istella | lane/apple-fast-gap-clus3 @ 2ac0505fc | clus3-sgdoc-simd-taxi, clus3-sgdoc-simd-istella | taxi 58,334 -> 42,252; istella 75,506 -> 106,443 | KEPT (small d only), FAST+Apple default for d <= 32 (`_OFF` off) | taxi -27.6%, fraction_flagged .05557 identical; istella +41% so d > 32 keeps the block form. The 0.8.34 time (~10 s istella) was the HOST CPU route (`_sgd_on_host`), not a GPU form; the istella regression stays open |
| `BAYES_GRID_GUARD` | bayesian-ridge / istella | lane/apple-fast-bayes @ 1a0bb2b2b | bayes-br-guard-istella | bayesian-ridge istella 2,630 (NaN) -> 3,040 (finite) | KEPT 6d4d55c99 | correctness: FAST grid path gave NaN |
| `CALIB_GNB_FOLDS` | calibrated / taxi | lane/apple-fast-meta @ 17b317ae6 | meta-calib-taxi | calibrated taxi 839.9 -> 49.8 | KEPT 24ed76679 | -94%; acc identical |
| `CD_FAST_GRID_GRAM` | elasticnet / istella; lasso / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-lasso-gg-istella, linear-enet-gg-istella | lasso istella 276 -> 201; enet 262 -> 196 | KEPT 0ae9c28cd | -27% / -25% |
| `CD_FAST_GRID_GRAM + CD_FAST_ROWMAJOR` | elasticnet / istella; lasso / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-lasso-rm-istella, linear-enet-rm-istella | lasso istella 264 -> 149; enet 262 -> 143 | KEPT 0ae9c28cd | -43% / -45% |
| `FAST_OLS_NORMAL_EQ (_OFF)` | ols / taxi, istella | lane/apple-fast-olsne @ 917dd5b4c | olsne-taxi, olsne-istella | taxi 286 -> 110; istella 2,921 -> 842 | KEPT 94cb5ba59 | -62% / -71%; resident fit gated to FAST |
| `LDAQDA_DEC_TILE` | qda / istella | lane/apple-fast-ldaqda @ 7d6a6126e | lq-* | all3 qda 15,794 -> 425 (RR alone 435) | KEPT b2dd5dfe5 |  |
| `LDAQDA_PAR_STAGES` | lda-clf / istella | lane/apple-fast-ldaqda @ 7d6a6126e | lq-* | alone 19,772 -> 19,723; all3 19,696 -> 536 | KEPT b2dd5dfe5 | helps only on top of RR_EIGH |
| `LDAQDA_RR_EIGH` | lda-clf, qda / istella | lane/apple-fast-ldaqda @ 7d6a6126e | lq-* | lda 19,374 -> 607; qda 15,855 -> 435 | KEPT b2dd5dfe5 | 32-36x; scoped to LDA/QDA in 7edf6d895 (it slowed IterativeImputer) |
| `MULTIOUT_RIDGE` | multioutput-reg / taxi | lane/apple-fast-meta @ 17b317ae6 | meta-mor-taxi | multioutput-reg taxi 163.0 -> 53.2 | KEPT 24ed76679 | -67%; r2 same |
| `QN_FAST_BLOCKS` | linearsvc / istella; linearsvr / istella; logreg / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-logreg-blocks-istella, linear-svc-blocks-istella, linear-svr-blocks-istella | logreg istella 3,893 -> 3,612; linearsvc 773 -> 733; linearsvr 913 -> 219 | KEPT 0ae9c28cd | -7% / -5% / -76% |
| `RIDGE_NO_U` | ridge / taxi, istella | lane/apple-fast-ridgespeed @ be56e2ead | ridgespeed-nou-* | on RESIDENT: istella 729 -> 542; taxi 21.7 -> 22.7 | KEPT 2caff7b0c | FAST-only bits |
| `RIDGE_RESIDENT` | ridge / taxi, istella | lane/apple-fast-ridgespeed @ be56e2ead | ridgespeed-res-* | taxi 57.1 -> 24.5; istella 1,405 -> 724 | KEPT 2caff7b0c | bit-identical route |
| `X_LINEAR_ENETCV_FAST (grid path)` | lasso-cv, enet-cv / taxi, istella | lane/apple-fast @ 952422579 | afc-enet2 | lasso-cv istella 70,824 -> 563; enet-cv istella 75,080 -> 578; taxi 3,100 -> 103 | KEPT 269ffa57a | 126-130x; r2 istella .310/.317 -> .326/.327 |
| `X_LINEAR_GRAM_SSE` | ard, bayesian-ridge / taxi, istella | lane/apple-fast @ 269ffa57a | afc-bayes2 | ARD istella 7,140 -> 864; BR istella 2,342 -> 3,105 | KEPT 269ffa57a | BR istella NaN on the grid path; fixed by BAYES_GRID_GUARD (6d4d55c99) |
| `X_LINEAR_LARS_FAST_GRAM` | lars / taxi; lasso-lars / taxi | lane/apple-fast-gram @ 3d36a676a | gram-lars-taxi, gram-llars-taxi | lars taxi 59.1 -> 15.1; lasso-lars taxi 59.1 -> 14.2 | KEPT c5e1bbeb6 | -75%; r2 same (re-measured on fixed head c338b88dd) |
| `X_LINEAR_RIDGE_FAST_GRAM` | ridge-clf / taxi; ridge-cv / taxi | lane/apple-fast-gram @ 3d36a676a | gram-ridgeclf-taxi, gram-ridgecv-taxi | ridge-clf taxi 168 -> 122; ridge-cv taxi 3,614 -> 37 | KEPT c5e1bbeb6 | -27% / -99% |
| `X_PREP_CLASS_COV_GRID` | lda-clf / istella; qda / istella | lane/apple-fast-gram @ 3d36a676a | gram-lda-istella, gram-qda-istella | lda istella 19,026 -> 19,641; qda istella 16,310 -> 15,811 | KEPT c5e1bbeb6 | kept for accuracy: lda acc .909 -> .913, qda .866 -> .881 |
| `KERNEL_FAST_BAYES_JACOBI` | bayesian-ridge / istella | lane/apple-fast-kernel @ 9e851777c | kernel-bayes-jacobi-ist | bayesian-ridge istella 2,741 -> 2,726 | DROPPED-noise | -0.6%; code removed from main e302f0ee8; recover at lane/apple-fast-kernel@9e851777c |
| `KERNEL_FAST_BAYES_STATS` | bayesian-ridge / istella | lane/apple-fast-kernel @ 9e851777c | kernel-bayes-stats-ist | bayesian-ridge istella 2,748 -> 3,143 | DROPPED-slower | +14.4%; define removed from main e33bb66e0 (code kept: KEPT BAYES_CLS1_STATS takes it); recover at lane/apple-fast-kernel@9e851777c |
| `OLS_FAST_DEVICE_CENTER` | ols / taxi | lane/apple-fast-core @ 9a31ebb4c | core-ols-dcenter-taxi | ols taxi 335.6 -> 157.6 | DROPPED-semantics | stale base: main OLS route changed (TSQR, then normal eq); code removed from main dd83da010; recover at lane/apple-fast-core@9a31ebb4c |
| `QN_FAST_COALESCED_OFF` | logreg / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-logreg-nocoal-istella | logreg istella 3,889 -> 4,996 | DROPPED-slower | +28.5% (turning coalescing off) |
| `QN_FAST_GRID_SUMS` | linearsvc / istella; linearsvr / istella; logreg / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-logreg-gs-istella, linear-svc-gs-istella, linear-svr-gs-istella | logreg 3,890 -> 4,235; svc 775 -> 802; svr 914 -> 213 | DROPPED-noise | mixed signs; code removed from main 5f1e86fd0; recover at lane/apple-fast-linear@1c7c213f8 |
| `RIDGE_FAST_CLS1_CODES + RIDGE_FAST_CLS1_PREDICT` | ridge-clf / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-rcall-taxi | ridge-clf taxi 120 -> 20.2 | DROPPED-noise | no better than CODES alone |
| `RIDGE_FAST_CLS1_PREDICT` | ridge-clf / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-rcpred-taxi | ridge-clf taxi -1.4% | DROPPED-noise | <5% |
| (baseline, no switch) | - | lane/apple-fast-robust @ cfdb95e48 | robust-ocsvm-base-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ARD_FAST_CLS1_BATCH` | ard / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ardbatch-taxi | ard taxi +21% alone | OPEN | slower alone, but kept within all three; merge pending |
| `ARD_FAST_CLS1_BATCH + ARD_FAST_CLS1_PARTS + ARD_FAST_CLS1_STATS` | ard / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ardall-taxi | ard taxi 14.4 -> 8.3 | OPEN | judged KEEP (-42.5%); merge pending |
| `ARD_FAST_CLS1_PARTS` | ard / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ardparts-taxi | ard taxi 14.3 -> 12.0 | OPEN | judged KEEP (-16%); merge pending |
| `ARD_FAST_CLS1_STATS` | ard / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ardstats-taxi | ard taxi 19.4 -> 13.6 | OPEN | judged KEEP (-30%); merge pending |
| `BAYES_FAST_CLS1_BATCH` | bayesian-ridge / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-brbatch-taxi | bayesian-ridge taxi -4% | OPEN | kept as part of all three (judged KEEP); merge pending |
| `BAYES_FAST_CLS1_BATCH + BAYES_FAST_CLS1_PARTS + BAYES_FAST_CLS1_STATS` | bayesian-ridge / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-brall-taxi | bayesian-ridge taxi 102 -> 15.3 | OPEN | judged KEEP (-85%, r2 same); merge pending |
| `BAYES_FAST_CLS1_PARTS` | bayesian-ridge / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-brparts-taxi | bayesian-ridge taxi -3% | OPEN | kept as part of all three (judged KEEP); merge pending |
| `BAYES_FAST_CLS1_STATS` | bayesian-ridge / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-brstats-taxi | bayesian-ridge taxi 99 -> 21.6 | OPEN | judged KEEP (-78%); cls1 merge pending |
| `HUBER_DEVICE_LBFGS` | huber / taxi | lane/apple-fast-robust @ cfdb95e48 | robust-huber-dev-taxi | huber taxi 242 -> 225 | OPEN | candidate (-7%, n=1); merge waits on the other robust lines |
| `HUBER_DEVICE_LBFGS + HUBER_FAST_BLOCK512` | huber / taxi | lane/apple-fast-robust @ cfdb95e48 | robust-huber-blk-taxi | huber taxi 226 -> 196 | OPEN | candidate (-13%); not merged yet |
| `LSVR_ALL` | linearsvr / taxi; linearsvr / istella | lane/apple-fast-linsvr @ c649076a4 (via lane/apple-fast-m2b1 65c454c87) | M3 linearsvr taxi; M3 istella (M2: linsvr-all-taxi-x 292.2 -> 58.6) | M3 taxi 117.2 -> 27.0; M3 istella +6.2% | KEEP for n_features <= 32 (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | -77% on taxi (d ~ 11), r2 / rmse identical; slower on istella (d ~ 220), so `QN_ALL_MAX_D = 32` gates the tiled objective and the device convergence (`qn_tiled_applies`, `dconv_applies`); wider data keeps LS_BATCH + EVAL_SLIM; `-D MOJOLEARN_LSVR_ALL_OFF` reverts |
| `LSVR_DEVICE_CONVERGE` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dconv-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dconv-vs-batch-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_DUAL_CD` | linearsvr / istella; linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dualcd-taxi, linsvr-dualcd-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_EVAL_SLIM` | linearsvr / taxi; linearsvr / istella | lane/apple-fast-linsvr @ c649076a4 | M3 taxi; linsvr-slim-istella-x (M2) | M3 taxi -17.4%; M2 istella -4.5% | KEPT lane/apple-fast-m2b1-main | quality identical; default, `-D MOJOLEARN_LSVR_EVAL_SLIM_OFF` reverts |
| `LSVR_FASTPATH_FIX` | linearsvr / taxi; linearsvr / istella | lane/apple-fast-linsvr @ c649076a4 | linsvr-fix-taxi-x (M2); M3 istella | M2 taxi -0.2%; M3 istella +1.5% | DROPPED-slower | not merged |
| `LSVR_FUSED_GRAD` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | M3 taxi | -1.9% | DROPPED-noise | not merged |
| `LSVR_LINESEARCH_BATCH` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 (via lane/apple-fast-m2b1) | M3 linearsvr taxi A/B (M2: linsvr-lsbatch-taxi-x -38.8%) | 116.7 -> 79.4 | KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | -32%; r2 / rmse identical; carries the fused-gradient pass (QN_FAST_FUSED) with it; `-D MOJOLEARN_LSVR_LINESEARCH_BATCH_OFF` reverts |
| `NB_CAT_ATOMIC` | categorical-nb / taxi | lane/apple-fast-nb @ be2ea3a05 | nb-cat-atomic-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `NB_TEXT_CSR` | multinomial-nb / text | lane/apple-fast-nb @ be2ea3a05 (via lane/apple-fast-m2b1) | M3 multinomial-nb text A/B (M2: nb-mnb-csr-text-x 221.7 -> 47.7) | 186.9 -> 35.4 | KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | -81%; accuracy .9831 / logloss .5595 identical; `-D MOJOLEARN_NB_TEXT_CSR_OFF` reverts |
| `RIDGE_FAST_CLS1_CODES` | ridge-clf / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-rccodes-taxi | ridge-clf taxi 120 -> 19.0 | OPEN | judged KEEP (-84%); merge pending |
| `ISOTONIC_FAST_NOLIST` (env `MOJOLEARN_ISOTONIC_FAST_NOLIST_OFF=1` off) | isotonic / istella | lane/apple-fast-gap-manprep @ 1db219f01 | gmp-iso-nolist-istella | isotonic istella 50.9 -> 29.4 | KEPT (FAST+Apple default) | -42%; r2/rmse same; fit's 3 + 2n output words were `tolist()`ed (2,000,000 Python floats at 1M rows), sliced and rebuilt; now three byte copies; same words |

## Neighbors (42)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `KDE_FAST_SLICES` | kde / istella | lane/apple-fast-core @ 9a31ebb4c | core-kde-slices-istella | kde istella 1,519 -> 141 | KEPT 857fd5804 | -91%; log-likelihood same |
| `KNN_FAST_MMA_BIGD` | knn-clf, knn / istella | lane/apple-fast @ 269ffa57a | afc-knn2 | knn-clf 864 -> 290; kneighbors 1,727 -> 1,788 | KEPT 269ffa57a | knn-clf 3.0x; kneighbors noise at n=1 |
| `KNN_FAST_MMA_K64` | knn / istella | lane/apple-fast-core @ 9a31ebb4c | core-knn-k64-istella | knn istella 1,581 -> 352 | KEPT 857fd5804 | -78%; recall same |
| `LP_FAST_RESIDENT` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-lp-res-taxi, n2-ls-res-taxi | label-propagation taxi 6,793 -> 2,086; label-spreading 2,095 -> 2,004 | KEPT 1e288d323 | -69% (comptime default) |
| `KNN_FAST_CLS1_PRESEED` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnseed-istella | knn istella 357 -> 343 | DROPPED-noise | -3.9%, n=1, marginal |
| `KNN_FAST_CLS1_PRESEED + KNN_FAST_CLS1_SLICES2` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnall-istella | knn istella 357 -> 385 | DROPPED-slower | slower |
| `KNN_FAST_CLS1_SLICES2` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnsl2-istella | knn istella 357 -> 421 | DROPPED-slower | slower |
| `LLE_FAST_KNN` | - | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-lle-knn-taxi; M3 re-check (lane/apple-fast-m2b1 batch) | lle taxi 4,266 -> 4,289; M3 -0.1% | DROPPED-noise | +0.5%; M2 ik-lle-knn-taxi-b +0.4%; M3 re-check -0.1% (inside spread): loser, never merges |
| `NC_FAST_CLS1_LABELS + NC_FAST_CLS1_PREDICT` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ncall-taxi | nearest-centroid taxi 105 -> 27 | DROPPED-noise | worse than LABELS alone |
| `NC_FAST_CLS1_PREDICT` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ncpred-taxi | nearest-centroid taxi -4% | DROPPED-noise | <5% |
| `RADIUS_FAST_REUSE_COUNT` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-radius-reuse-taxi | radius-neighbors taxi 0.1 -> 0.1 | DROPPED-noise | ms-scale; code removed from main f9af6028e; recover at lane/apple-fast-neighbors2@5fb6edd3f |
| (baseline, no switch) | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-k64chk-istella | knn istella (main) 357 | OPEN | baseline re-time only (K64 default); row flipped faster than sklearn 566 |
| `ANN_FAST_KNN_BIGD` | cagra / istella | lane/apple-fast-ann @ 70833546a | ann-cagra-knnbigd-istella | - | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_TEAM` | cagra / istella | lane/apple-fast-ann @ 70833546a | ann-cagra-team-istella | - | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_WIDE` | cagra / istella | lane/apple-fast-gap-cagra @ a3ebfc4a7 | gapcagra-wide-istella | cagra istella 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_DOT` | cagra / istella | lane/apple-fast-gap-cagra @ a3ebfc4a7 | gapcagra-dot-istella | cagra istella 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_IVFG` | cagra / istella | lane/apple-fast-gap-cagra @ a3ebfc4a7 | gapcagra-ivfg-istella | cagra istella 21,239 -> 1,294 | DROPPED-quality | recall .9838 -> .9595 |
| `CAGRA_FAST_IVFG + CAGRA_FAST_IVFG_P32` | cagra / istella | lane/apple-fast-gap-cagra @ a3ebfc4a7 | gapcagra-ivfg32-istella | cagra istella 21,239 -> 1,793 | DROPPED-quality | recall .9597; probes are not the loss |
| `CAGRA_FAST_SEEDS` | cagra / taxi | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-seeds-taxi | 2,887 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_SEEDS4` | cagra / taxi | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-seeds4-taxi | 2,887 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_ITERS` | cagra / taxi | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-iters-taxi | 2,887 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_SEEDS + CAGRA_FAST_ITERS` | cagra / taxi, istella | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-seedsiters-{taxi,istella} | 2,887 / 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_IVFG + CAGRA_FAST_IVFG_EXACTD` | cagra / istella | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-ivfgx-istella | 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_IVFG + IVFG_EXACTD + IVFG_P8` | cagra / istella | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-ivfgx8-istella | 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_IVFG + IVFG_EXACTD + SEEDS + ITERS` | cagra / istella | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-ivfgxsi-istella | 21,239 -> ? | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_IVFG_LOWD` | cagra / taxi (istella must be identical) | lane/apple-fast-w2-cagra (base b2b1c22bc) | w2-cagra-lowd-q, w2-cagra-lowd-taxi | cagra taxi 2,900 -> ? | OPEN | IVFG graph for d <= 64 (taxi d = 11 still built the exact 1.6e11-pair graph); gate recall@10 B >= A (tools/cagra_lowd_pair.py) |
| `CAGRA_FAST_IVFG_LOWD_SEEDS4` | cagra / taxi, istella | lane/apple-fast-w2-cagra (base b2b1c22bc) | w2-cagra-lowds4-q, w2-cagra-lowds4-taxi | M3 one run per arm: cagra taxi build 2,879.2 -> 714.9 ms (faiss-cpu 1,015.6) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF` | LOWD + SEEDS4 (search only); w2-cagra-lowds4-q PASS (recall@10 B >= A on taxi and istella). Plain LOWD stays opt-in (failed recall) |
| `ISOTONIC_FAST_PAIRMERGE + ISOTONIC_FAST_PAR` | isotonic / istella | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-iso-pair-istella-b (M2) | PAR alone timed out -> 165.5 | HELD (M2 only) | r2 .188; owes an M3 A/B; ported on lane/apple-fast-m2b1 |
| `ISOTONIC_FAST_PAR` | isotonic / istella | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-iso-par-istella | - | OPEN | A/B queued, no judged result yet |
| `IVFPQ_FAST_DEVICE_CODEBOOKS` | ivf-filter / istella; ivf-pq / istella; ivf-refine / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-devcb-istella, ann-ivfrefine-devcb-istella, ann-ivffilter-devcb-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_COARSE_RANDOM_INIT` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-rinit-istella, vsearch-ivf-rinit-istella, vsearch-ivf-rinit-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_DEVICE_VALIDATE` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-dval-istella, vsearch-ivf-dval-istella, vsearch-ivf-dval-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_FAST_DEVICE_CSR` | ivf / istella; ivf-pq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-csr-istella, ann-ivf-csr-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_FAST_DEVICE_TRAINSET` | ivf / istella; ivf-pq / istella; ivf-sq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-trainset-istella, ann-ivfsq-trainset-istella, ann-ivf-trainset-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_FAST_SCAN_SELECT` | ivf-pq / istella; ivf-rabitq / istella; ivf-sq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-select-istella, ann-ivfsq-select-istella, ann-ivfrq-select-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_KMEANS_LAZY_SHIFT` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-lazy-istella, vsearch-ivf-lazy-istella, vsearch-ivf-lazy-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_REFINE_TEAM` | ivf-refine / istella; ivf-refine / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-refine-team-istella, vsearch-refine-team-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE2_ALL` | kde / istella; kde / taxi | lane/apple-fast-batch @ 3150d75c1 | kde2-all-istella-x, kde2-all-vs-dimtile-istella-x | istella 139.5 -> 63.1; vs DIMTILE alone -0.6% | DROP | includes SAMPLE_FUSED (taxi +354%); the gain is DIMTILE |
| `KDE2_ALL + KDE_DIMTILE` | kde / istella | lane/apple-fast-kde2 @ 659400b94 | kde2-all-vs-dimtile-istella | taxi ~10 ms, jitter-dominated (lane/apple-fast-batch) | DROP | inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `KDE_DIMTILE` | kde / istella; kde / taxi | lane/apple-fast-batchv @ c8251211d | batchv-kde-dimtile-istella3, batchv-kde-dimtile-taxi3 | istella 137.2 -> 69.2 (-50%); taxi 9.0 -> 17.8 (+98%, ~10 ms scale) | KEEP for n_features > 32 (FAST + Apple default, _OFF) | mean_log_likelihood -222.27058 both; quality score_samples <= 9e-8 of scale (M2); d <= 32 keeps main fused pass |
| `KDE_DIMTILE + KDE_KERNEL_VARIANTS` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-variants-istella, kde2-variants-taxi | taxi ~10 ms, jitter-dominated (lane/apple-fast-batch) | DROP | inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `KDE_DIMTILE + KDE_LSE_FUSED` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-lse-istella, kde2-lse-taxi | taxi ~10 ms, jitter-dominated (lane/apple-fast-batch) | DROP | inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `KDE_DIMTILE + KDE_NORM_FUSED` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-norm-istella, kde2-norm-taxi | taxi ~10 ms, jitter-dominated (lane/apple-fast-batch) | DROP | inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `KDE_DIMTILE + KDE_SAMPLE_FUSED` | kde / istella | lane/apple-fast-kde2 @ 659400b94 | kde2-sample-ontile-istella | taxi ~10 ms, jitter-dominated (lane/apple-fast-batch) | DROP | inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `KDE_SAMPLE_FUSED` | kde / istella; kde / taxi | lane/apple-fast-batch @ 3150d75c1 | kde2-sample-taxi-x | taxi +354% | DROP | opt-in only |
| `NC_FAST_CLS1_LABELS` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-nclabels-taxi | nearest-centroid taxi 105 -> 21 | OPEN | judged KEEP (-80%, acc same); cls1 merge pending |
| `PQ_LUT_TILED` | ivf-pq / istella; ivf-pq / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-lut-istella, vsearch-pq-lut-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PQ_SCAN_FUSED` | ivf-filter / istella; ivf-pq / istella; ivf-pq / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-fused-istella, vsearch-filter-fused-istella, vsearch-pq-fused-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `TSNE_FAST_SPLIT` | tsne / istella | lane/apple-fast-ann @ 70833546a | ann-tsne-split-istella | - | OPEN | A/B queued, no judged result yet |
| `VSEARCH_ALL` | ivf / istella; ivf / taxi; ivf-filter / istella; ivf-filter / taxi; ivf-pq / istella; i... | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-all-istella, vsearch-ivf-all-istella, vsearch-ivf-all-taxi (+8) | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `XN_FAST_IMPUTE_TILED2` | knn-imputer / taxi | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-imp-t2-taxi-b (M2); M3 re-check | M2 +1.5%; M3 +5.6% | DROPPED-slower | slower on the M2 and the M3; never merged to main or lane/apple-fast-m2b1 (code only on its lane branch) |
| `XN_FAST_MMA_ROUTE` | lof / taxi; lle / taxi | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-lof-mma-taxi-b, ik-lle-mma-taxi-b (M2) | lof taxi 120.7 -> 2,659 (22x slower); lle -1.8% | DROPPED-slower | never on main; not merged |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_RES | ocsvm / taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-res-ocsvm-taxi | -77% alone | KEEP, FAST+Apple default (`_OFF` off) | Gram formed and solved on the device, no 400 MB round trip; quality identical (n=1) |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_2L | ocsvm / taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-2l-ocsvm-taxi | -5% alone | KEEP, FAST+Apple default (`_OFF` off) | two launches per SMO iteration; same alpha bits |
| MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256 | ocsvm / taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-{chunk256,all3}-ocsvm-taxi | 0% alone; all three 375.7 -> 65.0 | KEEP, FAST+Apple default (`_OFF` off) | kept with the combined A/B winner (fewer synchronizes) |
| `XN_FAST_IMPUTE_TIE_MEAN + XN_FAST_NAN_COLMISS_ONLY` | knn-imputer / taxi | lane/apple-fast-gap-manprep @ 9130a81bc | gmp-imp-taxi | masked_rmse 6.152 -> 5.109 (sklearn 5.26); fit 2.3 -> 2.7 ms | KEPT (FAST+Apple default, `_OFF`) | quality fix; IDENTICAL keeps the lower-index rule. quality fix: masked_rmse 6.15 vs sklearn 5.26; taxi's discrete columns tie at the k-th distance and the lower-index rule took the earliest (January) donors; tie mean = the mean of all donors at D_k filling the remaining slots (the expectation of a uniform tie-break). Fit: column NaN counts only, no cell list (expect fit 2.3 -> ~1.5 ms) |

## Prep (42)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `PREP_FAST_MINMAX` | minmax-scaler / istella | lane/apple-fast-prep @ a11e43a5e | prep-minmax-istella | minmax-scaler istella 152 -> 106 | KEPT f419ea9f1 | -30%; output digest identical |
| `SELECT_D` | select-d / taxi-hourly | lane/apple-fast-select @ 99fad7a5d | sel-d-hourly | select-d taxi-hourly 7.8 -> 3.9 | KEPT e2bfb8422 | -50%; digest identical |
| `SELECT_FCLS` | select-f-classif / taxi | lane/apple-fast-select @ 99fad7a5d | sel-fcls-taxi | select-f-classif taxi 131.4 -> 17.5 | KEPT e2bfb8422 | -87% |
| `SELECT_FREG` | select-f-regression / taxi; select-r-regression / taxi | lane/apple-fast-select @ 99fad7a5d | sel-freg-taxi, sel-rreg-taxi | select-r-regression taxi 100.5 -> 10.1; select-f-regression 102.6 -> 15.0 | KEPT 4198d5a9c | -90% / -85% |
| `X_PREP_FAST_UNIQUE` | - | lane/apple-fast-prep @ a11e43a5e | prep-onehot-uniq-taxi, prep-ordinal-uniq-taxi | onehot taxi 69.7 -> 30.4; ordinal taxi 67.2 -> 25.3 | KEPT f419ea9f1 | -56% / -62% |
| `X_PREP_FAST_NONEG` | - | lane/apple-fast-prep @ a11e43a5e | prep-onehot-noneg-taxi, prep-ordinal-noneg-taxi | onehot -2.1%; ordinal -0.1% | DROPPED-noise | <5%; code removed from main 10a9ab8eb; recover at lane/apple-fast-prep@a11e43a5e |
| `CV_FAST_SLICE` | cross-val-score / taxi | lane/apple-fast-resample @ 50b96e795 | resample-cv-slice-taxi | - | OPEN | A/B queued, no judged result yet |
| `CV_FAST_TRUST_FOLDS` | cross-val-score / taxi | lane/apple-fast-resample @ 50b96e795 | resample-cv-trust-taxi | - | OPEN | A/B queued, no judged result yet |
| `MI_ALL` | select-mutual-info(-reg) / istella; taxi | lane/apple-fast-miv @ 514401169 | miv-reg-all-*, miv-clf-all-* | reg istella 46,430 -> 1,151, taxi 2,316 -> 81; clf istella 715 -> 552, taxi 203 -> 145 | DROP (quality) | includes MI_FAST_FOLDS, which fails the selected-set gate; opt-in only |
| `MI_CLF_RANKMAJOR` | select-mutual-info / istella; taxi | lane/apple-fast-miv @ 5e26b1008 | miv-clf-rank-istella, miv-clf-rank-taxi | istella 715.2 -> 645.2 (-9.8%); taxi 202.2 -> 196.4 (-2.9%) | KEEP (FAST + Apple default, _OFF) | vs main; quality tools/miv_quality.sh (M2, 30k x 48 tie-heavy, cont / 5-value / 2-class y): scores bit-identical |
| `MI_REG_TIES` | select-mutual-info-reg / istella; taxi | lane/apple-fast-miv @ 514401169 | miv-reg-ties-istella, miv-reg-ties-taxi (old base: mi-reg-ties-istella-x 70,202 -> 1,414) | istella 46,465 -> 1,340 (-97%); taxi 2,339 -> 160 (-93%) | KEEP (FAST + Apple default, _OFF) | vs main; main istella swings 46-70 s, gap 35x; quality tools/miv_quality.sh (M2): scores bit-identical, set identical |
| `MI_FAST_FOLDS` | select-mutual-info(-reg) / istella; taxi | lane/apple-fast-batch @ 3150d75c1 | mi-reg-folds-istella-x | -0.2% | DROP (speed + quality) | quality tools/miv_quality.sh (M2, 30k x 48 tie-heavy, cont / 5-value / 2-class y): scores move up to 3.2% of scale, selected set changes (sym diff 2); opt-in only |
| `MI_REG_RANKMAJOR + MI_REG_SORTCOUNT` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-rankmajor-istella, mi-reg-rankmajor-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_REG_SORTCOUNT` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-sortcount-istella, mi-reg-sortcount-taxi | - | NOT A DEFAULT ALONE | SORTCOUNT is on under MI_REG_TIES (default); RANKMAJOR not A/B-ed vs main; opt-in |
| `MI_REG_SORTCOUNT + MI_REG_TIES` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-ties-istella, mi-reg-ties-taxi | old base 70,202 -> 1,414 | KEEP (see MI_REG_TIES row) | vs main on lane/apple-fast-miv: MI_REG_TIES row |
| `MI_WORK` | - | lane/apple-fast-mi @ 6944ebb57 | mi-reg-work-istella, mi-reg-work-taxi, mi-clf-work-istella, mi-clf-work-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP2_FAST_EIGH_BLOCK` | iterative-imputer / taxi | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-eigh-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP3_LABELS` | label-binarizer / taxi | lane/apple-fast-prep3 @ ec65873e3 | prep3-lb-taxi-x (M2); M3 re-check | M2 +1.0%; M3 -1.9% | DROPPED-noise | under 5% at n=1 on the M3, signs mixed with the M2; never merged to main or lane/apple-fast-m2b1 (code only on its lane branch) |
| `PREP3_MAXABS` | maxabs-scaler / istella | lane/apple-fast-prep3 @ ec65873e3 (via lane/apple-fast-m2b1) | prep3-maxabs-istella-x-m3 (M2: prep3-maxabs-istella-x 133.0 -> 99.1) | 121.7 -> 104.2 | KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | -14.4%; output digest bit-identical (same max_abs_ / scale_ words); `-D MOJOLEARN_PREP3_MAXABS_OFF` reverts |
| `PREP3_SPLINE` | spline / istella | lane/apple-fast-prep3 @ ec65873e3 | prep3-spline-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP_FAST_CLS2_MINMAX_FUSED / _MINMAX_POOL` | minmax-scaler | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |
| `PTIMPUTE_ALL` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-batchv @ 77f1f5afb | batchv-pt-all-istella, batchv-pt-all-taxi | istella 2,258 -> 512 (-77%); taxi B 65.5 (main arm crashed: core/staged_download bug, fixed) | DROP (quality) | PT quality (tools/batchv_quality.sh + batchv_quality_sk.sh, M2, 100k x 220 / x 11): lambdas vs define-off max rel 9.5e-3 (abs 2.2e-3), transform 8.4e-4 of scale; vs sklearn float64 lambdas 6.3e-3 rel (main 5.7e-3): fails the 1e-4 gate; stays opt-in |
| `PT_COLBATCH` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-batchv @ 30aa43339 | batchv-pt-nospec-istella, batchv-pt-nospec-taxi2 (COLBATCH + FUSED_TRANSFORM + SI_ONEPASS) | istella 2,262 -> 425 (-81%); taxi 305 -> 54.5 (-82%) | DROP (quality) | PT quality (tools/batchv_quality.sh + batchv_quality_sk.sh, M2, 100k x 220 / x 11): lambdas vs define-off max rel 9.5e-3 (abs 2.2e-3), transform 8.4e-4 of scale; vs sklearn float64 lambdas 6.3e-3 rel (main 5.7e-3): fails the 1e-4 gate; the lambda shift comes from COLBATCH (the SI arm alone keeps lambdas exact); stays opt-in |
| `PT_COLBATCH + PT_SPEC` | power-transformer / istella | lane/apple-fast-batch @ 3150d75c1 | ptimpute-pt-spec-vs-colbatch-istella | 910 -> 1,012 (+11%) | DROP | slower than COLBATCH; on main PTIMPUTE_ALL 512 vs no-SPEC set 425 |
| `PT_FOLD_NOX` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-nox-istella, ptimpute-pt-nox-taxi | - | DROP | moot under COLBATCH (notes/ptimpute.md); opt-in only |
| `PT_FUSED_TRANSFORM` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-batchv | (in batchv-pt-nospec-*) | see PT_COLBATCH | DROP (quality, with COLBATCH) | not A/B-ed alone vs main; stays opt-in |
| `PT_SPEC` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-batch @ 3150d75c1 | ptimpute-pt-spec-istella | +10% vs COLBATCH | DROP | opt-in only |
| `RESAMPLE_FAST_GATHER` | resample / taxi | lane/apple-fast-resample @ 50b96e795 | resample-rs-gather-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_IDX_BULK` | resample / taxi | lane/apple-fast-resample @ 50b96e795 | resample-rs-idxbulk-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_ONE_FOLD` | bootstrap / taxi | lane/apple-fast-resample @ 50b96e795 | resample-boot-onefold-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_PERM_SELECT` | permutation-test / taxi | lane/apple-fast-resample @ 50b96e795 | resample-perm-select-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_RANK_SORT` | bootstrap / taxi | lane/apple-fast-resample @ 50b96e795 | resample-boot-rank-taxi | - | OPEN | A/B queued, no judged result yet |
| `SI_ONEPASS` | simple-imputer / istella; simple-imputer / taxi | lane/apple-fast-batchv @ 77f1f5afb | batchv-si-onepass-istella, batchv-si-onepass-taxi | istella 303.7 -> 273.7 (-9.9%); taxi 26.4 -> 21.0 (-20%) | KEEP (FAST + Apple default, _OFF) | vs main; quality tools/batchv_quality.sh (M2): median stats exact, mean stats <= 1.2e-7 abs (1 ulp), PT lambdas exact |
| `X_PREP_FAST_CLS2_PACK / _PRESENT` | onehot, ordinal | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |
| `X_PREP_FAST_II_CONV` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-conv-taxi | both arms status=error | OPEN | env-form line; define-form -b relaunch pending |
| `X_PREP_FAST_II_GRAM_TILE` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-gram-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_QSELECT` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-si-qsel-istella, prep2-rs-qsel-istella | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_TE_ENC` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-te-enc-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_TE_GLOBAL` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-te-global-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |
| MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL | minmax-scaler / istella | lane/apple-fast-gap-cls2@72602a339 | gapcls2-pool-minmax-istella | 106 -> 21.9 | KEEP, FAST+Apple default (`_OFF` off) | pooled X buffer, no 880 MB allocation per fit; quality identical (n=1) |
| MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED | minmax-scaler / istella | lane/apple-fast-gap-cls2@72602a339 | gapcls2-{fused,fusedpool}-minmax-istella | -3% alone; 104.7 -> 19.1 with POOL | KEEP, FAST+Apple default (`_OFF` off) | NaN scan folded into the extrema pass; quality identical (n=1) |
| MOJOLEARN_X_PREP_FAST_CLS2_PACK | onehot, ordinal / taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-pack-{onehot,ordinal}-taxi | -19%, -33% | KEEP, FAST+Apple default (`_OFF` off) | distinct values packed into a small host region; quality identical (n=1) |
| MOJOLEARN_X_PREP_FAST_CLS2_PRESENT | onehot, ordinal / taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-present-{onehot,ordinal}-taxi | 32.2 -> 5.6, 28.2 -> 8.2 (with PACK) | KEEP, FAST+Apple default (`_OFF` off) | presence flags replace the sort for small integer columns; quality identical (n=1) |
| `X_PREP_FAST_STAGED_OUT` | label-binarizer / taxi; multilabel-binarizer / taxi; target-encoder / taxi | lane/apple-fast-gap-manprep @ 1169df581 | gmp-staged-lb-taxi, gmp-staged-mlb-taxi, gmp-staged-te-taxi | label-binarizer taxi 563 -> 345; multilabel-binarizer taxi 287 -> 193; target-encoder taxi 281 -> 272 | KEPT (FAST+Apple default, `_OFF`) | -38.7% / -33.0% / -3.2%; digests same; the program's output (LabelBinarizer's 1M x 259 int32 = 1 GB region; arena ranges of 1M+ words) downloaded through core/staged_download.mojo's pinned-stage pipeline instead of a raw host-pointer copy (~21 ms per 64 MB on Apple); copies only |
| `X_PREP_POOL_ARENA` | label-binarizer / taxi; multilabel-binarizer / taxi, istella | lane/apple-fast-w2-prep @ ec87a9035 | w2-pool-quality, w2-pool-* timing | label-binarizer taxi 319.9 -> 270.4; multilabel-binarizer taxi 175.6 -> 167.5; istella 129.3 -> 125.7 | DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_PREP_POOL_ARENA_OFF` | program device buffer (64 MB+) from core/device_pool.mojo instead of a fresh allocation (fresh-page memset cost); w2-pool-quality PASS, 25 output arrays sha256-identical, dirty-buffer repeats; copies and clears only, no bit moves; cost up to POOL_KEEP_BYTES (2 GB) idle device memory |

## Decomp (34)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `APPLE_FAST_GEMM_TN_V1` | ols | lane/apple-fast-tier @ 95a09d1fd | tier-pca-tnv1, tier-ols-tnv1 | ols istella 2,219 -> 935 | KEPT 7484503b6 | -58%; r2 .3211 -> .3319 (re-measured as tier-ols-tnv1b, define form) |
| `LDA_FUSED_SS` | lda / taxi-zones | lane/apple-fast-nb @ be2ea3a05 | nb-lda-fused-zones | lda taxi-zones 2,721 -> 1,509 | KEPT a7b8b9513 | -44.5%; perplexity same |
| `APPLE_FAST_GEMM_NT_TILED` | kmeans | lane/apple-fast-tier @ 95a09d1fd | tier-pca-nttiled, tier-ols-nttiled, tier-kmeans-nttiled | ols istella 2,179 -> 2,666; pca istella 599.6 -> 601.1; kmeans istella 1,474 -> 1,523 | DROPPED-slower | never wins; code removed from main 6f3e65746; recover at lane/apple-fast-tier@95a09d1fd |
| `DECOMP_FAST_SMALL_EIGH_J2` | fastica / istella | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-ica-istella | - | DROPPED-semantics | no-op after main removed _eigh2 |
| `LLE_SPARSE_EIG` | lle / taxi | lane/apple-fast-lle @ f2ea1ecb5 | lle-sparse-taxi | lle taxi 4,274 -> 9,174 | DROPPED-slower | +115% |
| `PCA_FAST_EIG` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-eig-istella, pca-eig-eig-taxi, pca-eig-tsvd-eig-istella (+2) | pca istella -2.2% / taxi +17%; tsvd 0%; ipca +0.4% / -1.2% | DROPPED-noise | mixed signs across datasets |
| `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` | pca / tsvd / ipca / istella | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-all-istella | pca istella -0.7% | DROPPED-noise | noise |
| `PCA_FAST_NO_ALIAS` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-noalias-istella, pca-eig-noalias-taxi | pca istella +2.1% / taxi -8.5% | DROPPED-noise | mixed signs |
| `PCA_FAST_TOPK` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-topk-istella, pca-eig-topk-taxi, pca-eig-tsvd-topk-istella | pca istella +1.5% / taxi -5.9%; tsvd +0.5% | DROPPED-noise | mixed signs |
| (baseline, no switch) | pca / tsvd / ipca | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-ident | - | OPEN | IDENTICAL baseline line |
| (baseline, no switch) | pca | lane/apple-fast-tier @ 95a09d1fd | tier-pca-ident, tier-rbf-ident, tier-theta-ident, tier-theta-fast | - | OPEN | IDENTICAL / FAST baseline lines |
| `LU_FAST_STEP1` | lu-factor / synthetic; lu-solve / synthetic | lane/apple-fast-gap-linalg2 @ f3d66dd94 | gl2-lu-step1-synthetic, gl2-lusolve-step1-synthetic | lu-factor synthetic 1,369 -> 977; lu-solve 1,422 -> 971 | KEPT | -28.6% / -31.7%; residual the same 3.256e-06; FAST+Apple default, -D MOJOLEARN_LU_FAST_STEP1_OFF reverts |
| `CHOL_FAST_DEVIO` | cholesky / synthetic | lane/apple-fast-gap-linalg2 @ 19fb05674 | gl2-chol-devio-synthetic | cholesky synthetic 435 -> 290 | KEPT | -33%; residual the same 1.659e-07; FAST+Apple default, -D MOJOLEARN_CHOL_FAST_DEVIO_OFF reverts |
| `CHOL_FAST_NOSYNC` | cholesky / synthetic | lane/apple-fast-gap-linalg2 @ 19fb05674 | gl2-chol-nosync-synthetic | cholesky synthetic 285 -> 271 (on top of DEVIO) | KEPT | -4.7%; residual the same; FAST+Apple default, -D MOJOLEARN_CHOL_FAST_NOSYNC_OFF reverts |
| `CHOL_FAST_TALL` | cholesky / synthetic | lane/apple-fast-w3-linalg @ 97b7bcb7e | w2-cholt-quality, w2-cholt-synthetic | cholesky synthetic 275.9 -> 260.8 ms; quality PASS | DEFAULT (FAST+Apple), rollback MOJOLEARN_CHOL_FAST_TALL_OFF | panel as one tall 64-step blocked factor (threadgroup-memory diag factor + inverse, matrix-unit in-place solve and panel update): drops the 256x256 inverse, its pack/unpack and ~19 of ~30 launches per panel; outer trailing update unchanged (differs from the DROPPED triangular-SYRK attempt, which changed only the trailing update). Quality: tools/chol_fast_tall_pair.py |
| `DECOMP_FAST_GEMM_MMA` | randomized-svd / istella, taxi; nmf / istella | lane/apple-fast-gap-linalg2-pca @ 474241154 | gl2p-rsvd-gemmmma-istella, gl2p-rsvd-gemmmma-taxi, gl2p-nmf-gemmmma-istella | randomized-svd istella 711 -> 533; taxi -1.3%; nmf istella 8,155 -> 6,333 | KEPT | -25% / -22%; reconstruction error the same (rsvd .0002359 / .0272, nmf .3252); FAST+Apple default for every x_decomp kit GEMM, -D MOJOLEARN_DECOMP_FAST_GEMM_MMA_OFF reverts |
| `PCA_FAST_GRAM_MMA` | pca / istella | lane/apple-fast-gap-linalg2-pca @ 474241154 | gl2p-pca-grammma-istella-r, gl2p-pca-quality (M2) | pca istella 606 -> 490 | KEPT | -19%; quality-only check: explained_variance_ rel diff 3.4e-06, subspace angle 1.8e-06 rad (nondeterministic MMA/atomic sum); FAST+Apple default, -D MOJOLEARN_PCA_FAST_GRAM_MMA_OFF reverts |
| `CHOL_FAST_BLOCKED` | cholesky / synthetic | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-chol-blocked-synthetic | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `CHOL_FAST_BLOCKED + SVD_FAST_CHOLQR` | svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-svd-cholqr-chol-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_DICT_UPDATE` | dict-learning / istella; mb-dict-learning / istella; mb-sparse-pca / istella; sparse-pc... | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-mbdl-upd-istella, dsp-dl-upd-istella, dsp-mbspca-upd-istella, dsp-spca-upd-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_GEMM_TILED` | als / taxi-zones; lstsq / istella; nmf / istella; randomized-svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-lstsq-tiled-istella, dlin-rsvd-tiled-istella, dlin-nmf-tiled-istella, dlin-als-tiled-taxizones | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_LASSO_BLOCK` | dict-learning / istella; mb-dict-learning / istella; sparse-pca / taxi | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-mbdl-lasso-istella, dsp-dl-lasso-istella, dsp-spca-lasso-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_DICT_DEV` | mb-dict-learning / istella | lane/apple-fast-gap-clus3 @ 43bc2906c | clus3-mbdl-dictdev-istella | 6,832 -> 5,193 | KEPT, FAST+Apple default (`_OFF` off) | -24%; sparsity .08636, recon err .6483 identical (n=1); `_update_dict` atom loop as 2 resident launches an atom (x_decomp/dict_fast.mojo), no per-atom vstack download / re-upload; resample + positive_dict keep the loop. Not decomp-sparse's DICT_UPDATE (never compiled) |
| `DECOMP_FAST_LASSO_GRP` | mb-dict-learning / istella | lane/apple-fast-gap-clus3 @ 43bc2906c | clus3-mbdl-lassogrp-istella | 6,827 -> 3,185 | KEPT, FAST+Apple default (`_OFF` off) | -53%; sparsity .08636, recon err .6483 identical (n=1); measured alone, combined number from the main retime; Lasso CD a 32-thread block per row, G/w/q/H in threadgroup memory (x_decomp/lasso_grp.mojo); was a thread per row (2 blocks at batch 256) with H in device memory; k <= 64 |
| `DECOMP_FAST_OMP_BLOCK` | sparse-coder / istella | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-omp-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_ALL` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-all-taxi, fa-all-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_EIG_SMALL + FA_ITER_DEVICE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-eig-taxi, fa-eig-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_EIG_SMALL + FA_ITER_DEVICE + FA_LL_DEVICE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-lldev-taxi, fa-lldev-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_FAST_QRR` | factor-analysis / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-fa-qrr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_GRAM_ONCE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-gram-taxi, fa-gram-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_ITER_DEVICE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-iter-taxi, fa-iter-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_ITER_DEVICE + FA_LIVEBUF` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-livebuf-taxi, fa-livebuf-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `FA_TRANSFORM_FUSED` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-tr-taxi, fa-tr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LU_FAST_PIVOT_GRID` | lu-factor / synthetic; lu-solve / synthetic | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-lu-pivot-synthetic, dlin-lusolve-pivot-synthetic | lu-factor synthetic 1,377 -> 1,233; lu-solve 1,367 -> 1,234 | OPEN | candidate (-10%); merge waits on the other decomp-linalg lines |
| `LU_FAST_MMA` | lu-factor / synthetic; lu-solve / synthetic | lane/apple-fast-w2-linalg @ 91a6573f5 (base 254e50a01) | w2-lumma-quality, w2-lumma-lufactor-synthetic, w2-lumma-lusolve-synthetic | 971.7 -> 849.3 (lu-factor); 971.9 -> 852.1 (lu-solve) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_LU_FAST_MMA_OFF`; w2-lumma-quality PASS (factor/solve residual gates, info, finite) | hypothesis: main's 256 scalar k=32 trailing passes (lu_trail_rb_kernel, ~46 GB traffic at n=8192) become 32 k=256 matrix-unit GEMMs (LAPACK getrf delayed update, 32-col inner panels unchanged via LU_FAST_STEP1); quality gate tools/lu_fast_mma_quality.py (factor + solve residual <= max(1.5x A, A+2e-7), info equal, board8192 bytes must differ) |
| `MCD_DEVICE_CSTEPS` | min-cov-det / taxi; elliptic-envelope / taxi | lane/apple-fast-robust @ cfdb95e48 | M3 min-cov-det taxi; robust-ee-taxi-x (M2) | M3 mcd 79,925 -> 215; M2 ee 64,578 -> 267.5 | DROPPED-quality (Oct 3; code kept opt-in `-D MOJOLEARN_MCD_DEVICE_CSTEPS` for a future correct parallel C-step) | tools/mcd_quality_ab.sh (M2, taxi 100k, mcdq4): Jaccard flagged Xq vs OFF .8805 mcd / .9645 ee (bar .99); OFF vs IDENTICAL .994 / .999; location_ 14%, covariance_ 18% rel Frobenius shift; mcd flag rate .231 -> .203; raw covariance rank 8 vs OFF/IDENTICAL 10 (all exact-fit singular) |
| `ANN3_COARSE_SEED + IVF_FAST_SEED_DEVICE` | ivf-pq / istella | lane/apple-fast-fastonly2 @ eca3e33b6 (via lane/apple-fast-m2b1) | fastonly2-5-ivf-pq-istella-m3, fastonly2-3-ivf-pq-istella-m3 | no seed -> seed+device 5,976 -> 5,253; host seed -> device seed 5,516 -> 5,243 | KEEP (FAST+Apple default, `_OFF`; lane/apple-fast-m2b1-m3) | -12.1% / -4.9%; recall_at_10 .5995 -> .6071 and .6017 -> .6071 (better); `-D MOJOLEARN_ANN3_COARSE_SEED_OFF` / `-D MOJOLEARN_IVF_FAST_SEED_DEVICE_OFF` revert (M2: 18,555 -> 12,036, -1.3%) |
| `QR_FAST_DEV` | qr / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-qr-dev-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SVD_FAST_CHOLQR` | svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-svd-cholqr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `XD_FAST_CLS2_GRP_DEVSCAN` | gaussian-rp / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | 54 -> 15.9 | OPEN | judged KEEP (-71%, keeps fit-time NaN refusal; beats sklearn 24.3); cls2 merge pending |
| `XD_FAST_CLS2_GRP_NOSCAN (+ _GRP_LAZY)` | gaussian-rp / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | 57 -> 1.1; with LAZY 54 -> 0.5 | OPEN | semantics: moves the NaN/inf error from fit to transform; Andrew asked, not default |
| MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN | gaussian-rp / istella, taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-devscan-grp-{istella,taxi} | 54.3 -> 15.9, 4.7 -> 3.9 | KEEP, FAST+Apple default (`_OFF` off) | device NaN scan replaces the one-thread host walk; quality identical (n=1) |
| MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN | gaussian-rp / istella, taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-noscan-grp-{istella,taxi} | 57 -> 1.1 (istella) | OPT-IN, pending Andrew | moves the NaN/inf refusal from fit to transform (semantics) |
| MOJOLEARN_XD_FAST_CLS2_GRP_LAZY | gaussian-rp / istella, taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-noscanlazy-grp-{istella,taxi} | 54 -> 0.5 (with NOSCAN) | OPT-IN, pending Andrew | measured only with NOSCAN; kept opt-in with it |
| `LLE_FAST_DEV_F0` (env `MOJOLEARN_LLE_FAST_DEV_F0_OFF=1` off) | lle / taxi, istella | lane/apple-fast-gap-manprep @ 9130a81bc | gmp-lle-devf0-taxi, gmp-lle-devf0-istella | lle taxi 3,827 -> 2,570; istella 3,895 -> 2,653 | KEPT (FAST+Apple default) | -33% / -32%; trustworthiness same; F0 = [F^ | u] built by three device cells instead of F.cols (400 MB download + 10,000 strided Python slices) and _hstack (F^ download, host move); same words for F0 |

## Cluster (38)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `GMM_FAST_BIG_CHOL` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-bc-istella | gmm istella 6,560 -> 5,894 | KEPT 0ae9c28cd | -10.2%; bic same |
| `MBK_ZEROCOPY` | minibatch-kmeans / istella, taxi | lane/apple-fast-mbkspeed @ 44e6d93e6 | mbkzc-istella, mbkzc-taxi | istella 351 -> 256; taxi 51.1 -> 48.5 | KEPT 51493c7a8 | -27% / -5% |
| `X_CLUSTER_FAST_MEANSHIFT` | meanshift / istella | lane/apple-fast-cluster @ f707846c8 | cluster-ms-istella | meanshift istella 52.1 -> 34.9 | KEPT 4b1311c12 | -33%; same clusters |
| `X_CLUSTER_FAST_MINIBATCH` | minibatch-kmeans / istella | lane/apple-fast-cluster @ f707846c8 | cluster-mb-istella | minibatch-kmeans istella 388 -> 352 | KEPT 4b1311c12 | -9.5%; silhouette .1167 -> .1182 |
| `GMM_FAST_BIG_CHOL + GMM_FAST_ESTEP_STACK + GMM_FAST_GRID_COV` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-all-istella | gmm istella 6,587 -> 9,046 | DROPPED-slower | +37.3%; GMM_FAST_ESTEP_STACK: code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8; GMM_FAST_GRID_COV: code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8 |
| `GMM_FAST_ESTEP_STACK` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-es-istella | gmm istella 6,566 -> 6,886 | DROPPED-slower | +4.9%; code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8 |
| `GMM_FAST_GRID_COV` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-gc-istella | gmm istella 6,584 -> 9,858 | DROPPED-slower | +49.7%; n_iter 24 -> 42; code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8 |
| `KMEANS_FAST_ROWNORM` | kmeans / taxi | lane/apple-fast-core @ 9a31ebb4c | core-kmeans-rownorm-taxi | kmeans taxi -3.1% | DROPPED-noise | <5% at n=1; inertia same; code removed from main 073bd0029; recover at lane/apple-fast-core@9a31ebb4c |
| `KMEANS_FAST_SKIP_PREDICT` | kmeans / taxi | lane/apple-fast-core @ 9a31ebb4c | core-kmeans-skippred-taxi | kmeans taxi -2.2% | DROPPED-noise | <5% at n=1; code removed from main a595988e8; recover at lane/apple-fast-core@9a31ebb4c |
| `AFFINITY_FAST_LOOP` | affinity-prop / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-ap-loop-istella | - | OPEN | A/B queued, no judged result yet |
| `AP_EXACT` | affinity-prop / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-ap-exact-istella | - | OPEN | A/B queued, no judged result yet |
| `AP_SPLIT` | affinity-prop / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-ap-split-istella | - | OPEN | A/B queued, no judged result yet |
| `BGMM_ENT` | bayesian-gmm / taxi | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bgmm-ent-taxi | - | OPEN | A/B queued, no judged result yet |
| `BGMM_ESTEP1` | bayesian-gmm / taxi | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bgmm-estep1-taxi | - | OPEN | A/B queued, no judged result yet |
| `BGMM_FAST_MAHAL_GEMM` | bayesian-gmm / taxi | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bgmm-mahal-taxi | - | OPEN | A/B queued, no judged result yet |
| `BGMM_FAST_MOMENTS_GEMM` | bayesian-gmm / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bgmm-momgemm-istella | - | OPEN | A/B queued, no judged result yet |
| `BISECT_FAST_RESIDENT` | bisecting-kmeans / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bisect-resident-istella | - | OPEN | A/B queued, no judged result yet |
| `CC_FAST` | connected-components / taxi | lane/apple-fast-graph @ 1fa36a7ec | graph-cc-fast-taxi | - | OPEN | A/B queued, no judged result yet |
| `DBSCAN_FAST_CC_BATCH` | dbscan / taxi | lane/apple-fast-core @ 9a31ebb4c | core-dbscan-ccbatch-taxi | dbscan taxi: both arms time out | OPEN | incomplete |
| `DBSCAN_FAST_DENSEBALL` | dbscan / taxi, istella | lane/apple-fast-dbscantaxi @ 1febff7df | dbscantaxi-ab, dbscantaxi-ab-ist | - | OPEN | queued; arm A times out on taxi; same-bits ID owed |
| `DBSCAN_FAST_SCAN` | dbscan / taxi | lane/apple-fast-core @ 9a31ebb4c | core-dbscan-scan-taxi | dbscan taxi: both arms time out | OPEN | incomplete; replaced by DBSCAN_FAST_DENSEBALL lane |
| `HDBSCAN2_ALL` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-all-taxi, hdbscan2-all6-taxi, hdbscan2-all-istella, hdbscan2-all6-istella | - | DROP (as a bundle) | its gain is HDB_SMR_TILED (KEEP); CORE_TILE and ONE_SYNC lose vs main; opt-in only |
| `HDB_CORE_TILE` | hdbscan / taxi; hdbscan / istella | lane/apple-fast-batchv @ c7ede6e47 | batchv-hdb-core-taxi, batchv-hdb-core-istella | taxi vs main +0.8%; istella 44,717 -> 45,590 (+2%) | DROP | old-base -11% did not carry to main; clusters identical; opt-in only |
| `HDB_DEV_BORUVKA` | hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-boruvka-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_LINKAGE_DEVICE` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-linkage-taxi, hdbscan2-linkage-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_ONE_SYNC` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-batchv @ c8251211d | batchv-hdb-onesync-taxi, batchv-hdb-onesync-istella | taxi 424.7 -> 425.5 (+0.2%); istella 3,809 -> 3,829 (+0.5%, SMR on both) | DROP | clusters identical; old-base -5% did not carry; opt-in only |
| `HDB_SELECT_DEVICE` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-select-taxi, hdbscan2-select-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_SMR_TILED` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-batchv @ c7ede6e47 | batchv-hdb-smr-istella, batchv-hdb-smr-taxi | istella 44,879 -> 3,800 (-91.5%); taxi 426 -> 434 (+2%, noise: taxi takes the d <= 64 arm) | KEEP (FAST + Apple default, _OFF) | vs main, one run per arm; n_clusters 47 / noise 0.25381 (istella), 160 / 0.14222 (taxi) identical both arms |
| `OPTICS2_ALL` | optics / istella; optics / taxi | lane/apple-fast-batch @ 3150d75c1 | optics2-all-taxi-x | taxi 411 -> 34,474 (84x slower) | DROP | includes STEP_BATCH; never merged |
| `OPTICS_CORE_SQ` | optics / istella; optics / taxi | lane/apple-fast-batch @ 3150d75c1 | optics2-sq-taxi-x | taxi +6.1% | DROP | never merged |
| `OPTICS_FAST_DEVICE_ORDER` | optics / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-optics-devorder-istella | - | OPEN | A/B queued, no judged result yet |
| `OPTICS_FRONTIER_DEVICE` | optics / istella; optics / taxi | lane/apple-fast-opv @ 9a844772e | opv-fd-istella, opv-fd-taxi (old base: optics2-fd-istella-x 433 -> 259) | vs main istella 272.8 -> 272.0 (-0.3%); taxi 253.3 -> 250.5 (-1.1%) | DROP | main's OPTICS_FAST_DEVICE_ORDER (cluster2) already took the gain; clusters/silhouette identical; never merged |
| `OPTICS_LIVEBUF` | optics / istella; optics / taxi | lane/apple-fast-opv @ 9a844772e | opv-lb-istella, opv-lb-taxi (old base -8.6%) | vs main istella 271.4 -> 405.8 (+50%); taxi 253.1 -> 383.4 (+52%) | DROP | turns on optics2's route, which bypasses main's faster device order; never merged |
| `OPTICS_STEP_BATCH` | optics / istella; optics / taxi | lane/apple-fast-batch @ 3150d75c1 | optics2-sb-taxi-x | taxi 416 -> 22,564 (54x slower) | DROP | never merged |
| `X_CLUSTER_FAST_CLS2_MBK_G128 / _MBK_FIN / _MBK_POOL` | minibatch-kmeans / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN | merge guard: never default MBK_FIN (one-block kernel) |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL | minibatch-kmeans / istella, taxi | lane/apple-fast-gap-cls2@72602a339 | gapcls2-pool-mbk-{istella,taxi} | 256.7 -> 170.7, 42.5 -> 38.1 | KEEP, FAST+Apple default (`_OFF` off) | pooled X buffer; quality identical (n=1) |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128 | minibatch-kmeans / istella, taxi | lane/apple-fast-gap-cls2@72602a339 (deleted before merge) | gapcls2-g128-mbk-{istella,taxi} | +14%, +22% | DROP, deleted before merge | slower |
| MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN | minibatch-kmeans / istella, taxi | lane/apple-fast-gap-cls2@72602a339 (deleted before merge) | gapcls2-fin-mbk-{istella,taxi} | +5%, -3% | DROP, deleted before merge | noise; a single-block kernel (no-one-block rule) |
| `RESAMPLE_FAST_IDX_DIRECT` | resample / taxi, istella | lane/apple-fast-gap-manprep @ c90b63b26 | gmp-rs-idx-taxi, gmp-rs-idx-istella | resample taxi 71.7 -> 62.8; istella -1.8% | KEPT (FAST+Apple default, `_OFF`) | -12.4%; max_mean_shift same; the 1M device row draws copied straight into the caller's int32 Array (was host buffer -> List append loop -> store loop); same integers. The numpy row gather stays (a device gather moves the whole matrix up and back) |
| `BISECT_FAST_ZEROCOPY` | bisecting-kmeans / istella | lane/apple-fast-gap-clus3 @ 43bc2906c | clus3-bisect-zc-istella | 2,033 -> 848 | KEPT, FAST+Apple default (`_OFF` off) | -58.3%; silhouette .1183 identical (n=1); X from the caller's pointer into a pooled buffer, column means + centering on the device (no host copy, no host mean/center passes over 220M values, no centered 880 MB list), inertia = sum of leaf scores (no second X upload, no n x k pass + download); unit weights + biggest_inertia only. Expect -25..-40% |
| `X_CLUSTER_FAST_CLS3_MBK_ROWGRP` | minibatch-kmeans / istella | lane/apple-fast-gap-clus3 @ 43bc2906c | clus3-mbk-rowgrp-istella | 172 -> 148 | KEPT, FAST+Apple default (`_OFF` off) | -14.0%; silhouette .1182 identical (n=1); batch assignment a 32-thread group per row (512 blocks, coalesced row reads) instead of a thread per row (16 blocks); k <= 16 |

## Time series (31)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `ARIMA_FAST_ASYNC` | autoarima / taxi-hourly, synthetic | lane/apple-fast-gap-arima @ d967c0121 | gaparima-async-* | taxi-hourly 39,639 -> 24,563; synthetic 28,511 -> 14,918 | KEPT c73305a4c | -38% / -48%; rmse identical |
| `ARIMA_FAST_EVAL_WS` | autoarima / taxi-hourly | lane/apple-fast-tsa @ cc25a4c7f | tsa-evalws-autoarima | autoarima taxi-hourly 50,299 -> 42,921 | KEPT 6c54b7e87 | -14.7% |
| `ARIMA_FAST_EVAL_WS + ARIMA_FAST_LLONLY` | autoarima / taxi-hourly | lane/apple-fast-tsa @ cc25a4c7f | tsa-both-autoarima | autoarima taxi-hourly 50,500 -> 39,504 | KEPT 6c54b7e87 | -21.8% |
| `ARIMA_FAST_LLONLY` | autoarima / taxi-hourly | lane/apple-fast-tsa @ cc25a4c7f | tsa-llonly-autoarima | autoarima taxi-hourly 50,459 -> 47,023 | KEPT 6c54b7e87 | -6.8%; rmse identical |
| `ETS_TEAM` | damped-ets / taxi-hourly | lane/apple-fast-ets @ 03b4948d3 | ets-team-taxi | damped-ets taxi-hourly 611.9 -> 48.0 | KEPT aec9f8c33 | -92%; rmse same |
| `GARCH_COOP` | garch / synthetic, taxi-hourly | lane/apple-fast-garchspeed @ d6effc734 | garchcoop-synthetic, garchcoop-taxi-hourly | synthetic 383 -> 15.1; taxi-hourly 582 -> 31.0 | KEPT 321a1a81c | 25x / 19x; llf higher |
| `PROPHET_COOP` | prophet / synthetic, taxi-hourly | lane/apple-fast-prophetspeed @ 1f75f9068 | ps-coop-synthetic, ps-coop-taxi-hourly | synthetic 431 -> 45; taxi-hourly 360 -> 43 | KEPT 8c2b7de51 | -89%; rmse same |
| `SELECT_D` | select-d / synthetic; select-d / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-seld-selectd-taxi-hourly, gaptsa-seld-selectd-synthetic | see select lane | KEPT e2bfb8422 | same define merged from lane/apple-fast-select |
| `SEQ_CROSTON_REG` | croston / synthetic; croston / taxi-hourly | lane/apple-fast-seq @ b185f069f | seq-croston-reg, seq-croston-reg-syn | croston taxi-hourly 3.0 -> 2.4; synthetic 2.7 -> 1.8 | KEPT 2dcdd949f | ms-scale; rmse same |
| `SEQ_FAST_FMA` | theta / taxi-hourly | lane/apple-fast-tier @ 95a09d1fd | tier-theta-fma | theta taxi-hourly (FMA alone, arm A failed) | KEPT d1fa9223b | default with THETA_REG; later cause of the dyn-opt-theta regression, fixed by SEQ_FAST_THETA_SNAP |
| `SEQ_FAST_FMA + SEQ_THETA_REG` | theta / taxi-hourly | lane/apple-fast-tier @ 95a09d1fd | tier-theta-regfma | theta taxi-hourly 1,769 -> 220 | KEPT d1fa9223b | -87.6%; rmse 49.28 -> 49.02 |
| `SEQ_FAST_THETA_SNAP` | dynamic-optimized-theta / taxi-hourly | lane/apple-fast-regress @ 93951fafb | regress-dotm-snap | 2,088 -> 562 | KEPT f6a9e7c04 | -73%, same bits; fixes the SEQ_FAST_FMA regression (FMA_OFF also fixed it: 2,089 -> 540) |
| `SEQ_FAST_THETA_SPEC` | theta / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-thetaspec-theta-taxi-hourly | theta taxi-hourly 218 -> 20.3 | KEPT 6c0ba4379 | -91%; quality identical; measured at bab8ef08f |
| `SEQ_FAST_VAR_ONECOPY` | var / synthetic; var / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-varonecopy-var-taxi-hourly, gaptsa-varonecopy-var-synthetic | var taxi-hourly 5.9 -> 5.1; synthetic 6.2 -> 5.0 | KEPT 6c0ba4379 | -13% / -19%; measured at 12c6407af |
| `SEQ_GARCH_GRID` | garch / taxi-hourly | lane/apple-fast-seq @ b185f069f | seq-garch-grid | garch taxi-hourly 2,769 -> 2,787 | KEPT 2dcdd949f | noise alone; merged with REG |
| `SEQ_GARCH_GRID + SEQ_GARCH_REG` | garch / synthetic; garch / taxi-hourly | lane/apple-fast-seq @ b185f069f | seq-garch-reggrid, seq-garch-reggrid-syn | garch taxi 2,769 -> 905; synthetic 1,525 -> 528 | KEPT 2dcdd949f | see SEQ_GARCH_REG note |
| `SEQ_GARCH_REG` | garch / taxi-hourly | lane/apple-fast-seq @ b185f069f | seq-garch-reg | garch taxi-hourly 2,770 -> 968 | KEPT 2dcdd949f | A/B base was stale: board path uses garch_team, so no main speedup (GARCH_COOP did it) |
| `SEQ_THETA_REG` | theta / taxi-hourly | lane/apple-fast-tier @ 95a09d1fd | tier-theta-reg | theta taxi-hourly 1,770 -> 1,326 | KEPT d1fa9223b | -25% |
| `TSA2_STL` | stl / taxi-hourly | lane/apple-fast-tsa2 @ b27c8169b | tsa2-stl-taxi-hourly | stl taxi-hourly 359.0 -> 7.0 | KEPT 550806bc0 | -98%; residual_std same |
| `TSA2_VAR` | var / taxi-hourly | lane/apple-fast-tsa2 @ b27c8169b | tsa2-var-taxi-hourly | var taxi-hourly 7.6 -> 5.0 | KEPT 550806bc0 | -34% |
| `TSA_FAST_KPSS_PACK` | kpss / synthetic; kpss / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-kpsspack-kpss-taxi-hourly, gaptsa-kpsspack-kpss-synthetic | kpss taxi-hourly 5.2 -> 1.5; synthetic 5.3 -> 2.0 | KEPT 6c0ba4379 | -72% / -62%; quality identical; measured at 44100699e |
| `TSA_FAST_SELD_FUSED` | select-d / synthetic; select-d / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-seldfused-selectd-taxi-hourly, gaptsa-seldfused-selectd-synthetic | select-d taxi-hourly 8.5 -> 2.0; synthetic 8.0 -> 2.2 | KEPT 6c0ba4379 | -76% / -73% (SELECT_D alone -49% / -39%); measured at 44100699e |
| `ARIMA_FAST_LS_NOREAD` | autoarima / synthetic | lane/apple-fast-gap-arima @ d967c0121 | gaparima-noread-synthetic | 28,578 -> 28,587 | DROPPED-noise | 0%; code removed from main 569e80238; recover at lane/apple-fast-gap-arima@d967c0121 |
| `ARIMA_FAST_P_FIX` | autoarima / taxi-hourly | lane/apple-fast-gap-arima @ d967c0121 | gaparima-pfix-taxi-b, gaparima-pfixasync | 39,550 -> 104,849; with ASYNC 39,570 -> 64,830 | DROPPED-slower | +165%; code removed from main dd47c5df0; recover at lane/apple-fast-gap-arima@d967c0121 |
| `SEQ_FAST_VAR_ONECOPY + SEQ_FAST_VAR_SPEC` | var / synthetic; var / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-varboth-var-taxi-hourly, gaptsa-varboth-var-synthetic | = VAR_ONECOPY alone | DROPPED-noise | SPEC adds nothing |
| `SEQ_FAST_VAR_SPEC` | var / synthetic; var / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-varspec-var-taxi-hourly, gaptsa-varspec-var-synthetic | var +14% / -2% | DROPPED-slower | deleted from main at merge; recoverable at 5057bee75 |
| `TSA2_KPSS` | kpss / synthetic; kpss / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-tsa2kpss-kpss-taxi-hourly, gaptsa-tsa2kpss-kpss-synthetic | kpss taxi-hourly -73%, synthetic -62% | DROPPED-noise | same gain as TSA_FAST_KPSS_PACK (kept); stays opt-in; code removed from main 7ff2caf99; recover at lane/apple-fast-gap-tsa@e9da47064 |
| `TSA2_KPSS` | kpss / taxi-hourly | lane/apple-fast-tsa2 @ b27c8169b | tsa2-kpss-taxi-hourly | compile fail on this branch; on gap-tsa: kpss taxi-hourly -73%, synthetic -62% | DROPPED-noise | same gain as TSA_FAST_KPSS_PACK, which was kept instead; stays opt-in; code removed from main 7ff2caf99; recover at lane/apple-fast-tsa2@b27c8169b |
| (baseline, no switch) | - | lane/apple-fast-seq @ b185f069f | seq-croston-ident, seq-croston-fast | - | OPEN | IDENTICAL / FAST baseline lines |
| `SEQ_FAST_THETA_HOIST` | theta / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-thetahoist-theta-taxi-hourly | theta taxi-hourly 218 -> 58 | DROPPED-noise | -73% alone (at 976a585a0); on top of THETA_SPEC (default) gaptsa-spechoist-theta-taxi-hourly 21.8 -> 21.5 (-1%, noise); code removed from main 239fde86d; recover at lane/apple-fast-gap-tsa@e9da47064 |
| `SEQ_GARCH_HOST_MAX` | - | lane/apple-fast-seq @ b185f069f | seq-garch-ident-dev, seq-garch-dev | - | OPEN | env-form line (no-op after define switch) |

## Kernel / GP (11)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `APPLE_FAST_GEMM_PINNED` | nys / taxi | lane/apple-fast-tier @ 95a09d1fd | tier-rbf-pinned, tier-rbf-pinned-taxi, tier-nys-pinned | rbf taxi 65.0 -> 63.9; nystroem istella 525.6 -> 522.7 | DROPPED-noise | -1.6% / -0.6%; code removed from main 75c59ea26; recover at lane/apple-fast-tier@95a09d1fd |
| `KAPPROX_DEVICE` | additive-chi2 / istella; skewed-chi2 / taxi | lane/apple-fast-kapprox @ 10d5a7970 | kap-schi2-taxi, kap-achi2-istella | skewed-chi2 taxi 2.8 -> 1.3; additive-chi2 istella 11.0 -> 12.5 | DROPPED-quality | kernel_rel_error .0378 -> .0480 (worse) / slower; code removed from main 9d5baaa9b; recover at lane/apple-fast-kapprox@10d5a7970 |
| `KERNEL_FAST_GPR_RESIDENT` | gpr / istella | lane/apple-fast-kernel @ 9e851777c | kernel-gpr-resident-ist | gpr istella 168 -> 133 | DROPPED-semantics | not made default: main resident GPR chain already covers it; lane arm folded on host; note removed from main cc300add8 (no code was on main); recover at lane/apple-fast-kernel@9e851777c |
| `SPARSE_RP_DEVICE` | sparse-rp / taxi | lane/apple-fast-kapprox @ 10d5a7970 | kap-srp-taxi | sparse-rp taxi 4.9 -> 1.2 | DROPPED-quality | mean_abs_distortion .147 -> .236 (worse); code removed from main e58326562; recover at lane/apple-fast-kapprox@10d5a7970 |
| `SVGP_FAST_GPU` | svgp / taxi | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-svgp-gpu-taxi | svgp taxi 924 -> 903 | DROPPED-noise | -2.3% n=1; opt-in removed at merge |
| `XN_FAST_TILED_RBF` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-ocsvm-tiled-istella | ocsvm istella 440 -> 445 | DROPPED-noise | +1.2%; code removed from main f103d7381; recover at lane/apple-fast-neighbors2@5fb6edd3f |
| `XN_PCS_SPARSE` | poly-count-sketch / taxi | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-pcs-sparse-taxi | poly-count-sketch taxi 0.3 -> 0.3 | DROPPED-noise | no change; code removed from main 3d1c6bd73; recover at lane/apple-fast-neighbors2@5fb6edd3f |
| (baseline, no switch) | - | lane/apple-fast-kapprox @ 10d5a7970 | kap-grp-base-taxi, kap-grp-base-istella | - | OPEN | baseline lines |
| `KERNEL_FAST_NYS_RR_EIGH` | nystroem / istella | lane/apple-fast-kernel @ 9e851777c | kernel-nys-rr-ist | - | OPEN | A/B queued, no judged result yet |
| `KPCA_RESIDENT` | kernel-pca / istella | lane/apple-fast-kapprox @ 10d5a7970 | kap-kpca-istella | kernel-pca istella 1,014 -> 294 | OPEN | digests differ; quality vs sklearn not yet checked |
| `XN_FAST_CLS2_OCSVM_RES / _2L / _CHUNK256` | ocsvm | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |

## Neural (99)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `MOE_DEVGROUP` | moe / synthetic | lane/apple-fast-moespeed @ b9bc99e0f | moespeed-* | 73.4 -> 71.9 | KEPT 499e1b72c | removes the host sort round trip (GPU-only rule) |
| `MOE_REGTILE` | moe / synthetic | lane/apple-fast-moespeed @ b9bc99e0f | moespeed-* | 789 -> 72.7 | KEPT 499e1b72c | 10.9x; digest identical |
| `OPT_PIPE_DOWN` | adagrad, lamb, adamax / synthetic | lane/apple-fast-optspeed @ 7a5b3fb2c | optspeed-* | adagrad 318 -> 191 | KEPT 5990c5946 | -40%; digests identical |
| `OPT_ZERO_OPEN` | adagrad, lamb, adamax / synthetic | lane/apple-fast-optspeed @ 7a5b3fb2c | optspeed-* | all3 adagrad 317 -> 187; lamb 340 -> 206; adamax 326 -> 195 | KEPT 5990c5946 | copy only, no bit change |
| `SEQ_FAST_PIPE_DOWN` | adafactor, layernorm, adagrad / synthetic | lane/apple-fast-regress @ 93951fafb | regress-adafactor-down, regress-adafactor-pipe, regress-adagrad-pipe | adafactor 684 -> 390; adagrad 195.5 -> 199.1 | KEPT f6a9e7c04 | -43% adafactor; adagrad neutral |
| `SEQ_FAST_PIPE_UP` | layernorm / synthetic | lane/apple-fast-regress @ 93951fafb | regress-layernorm-pipe | 76.0 -> 52.2 (with PIPE_DOWN) | KEPT f6a9e7c04 | helps layernorm, neutral elsewhere |
| `OPT_RAW_UP` | adagrad / synthetic | lane/apple-fast-optspeed @ 7a5b3fb2c | optspeed-* | 310 -> 307 | DROPPED-noise | stays opt-in |
| (baseline, no switch) | neural | lane/apple-fast-neural @ 600237d7c | afn-tier-base-gemm, afn-tier-base-gemm-bf16, afn-tier-base-gemm-int8 (+9) | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-all, afn-w2-lmgrad-fwd-attnall-forward | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-arena | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_FLASH` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-flash | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_FUSE_MLP` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-fuse-mlp, afn-w2-lmgrad-fwd-fusemlp-forward | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_FUSE_PRE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-fuse-pre | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_GQA_TILE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-gqa-tile | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_NORM_SG` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-norm-sg | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_ATTN_ROPE_CACHE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-rope-cache | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_CNN_DIRECT` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-cnn-direct | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_EMB_ATOMIC_BWD` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-emb-atomic | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_EPI_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-m1-all, afn-w2-epi-m2-all, afn-w2-epi-m3-all, afn-w2-epi-samba-all-mamba-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_EPI_ALL + AFN_SAMBA_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-all-training-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM2_ALL + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-all, afn-w2-gemm2-gemm-bf16-all, afn-w2-gemm2-transformer-forward-all | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM2_BIGTILE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-bigtile, afn-w2-gemm2-gemm-bf16-bigtile | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM2_DBUF + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-dbuf, afn-w2-gemm2-gemm-bf16-dbuf | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM2_DIRECT_B + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-direct-b, afn-w2-gemm2-gemm-bf16-direct-b | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM2_SWIZZLE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-swizzle, afn-w2-gemm2-gemm-bf16-swizzle | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-all, afn-gemm-bf16-all, afn-gemm-int8-all | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_BF16_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-bf16-bf16mma | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_INT8_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-int8-int8mma | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-simdgroup, afn-gemm-bf16-simdgroup | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_SIMDGROUP + AFN_GEMM_SPLITK + AFN_LM_WGRAD_SPLIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-wgrad-vs-gemmsplitk-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_SIMDGROUP + AFN_LMGRAD_ALL + AFN_LM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-all-on-stack-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_SPLITK` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-splitk | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_TILESHAPE` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-tileshape | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LMGRAD_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-all-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-all-train, afn-lm-all-forward | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_BWD_EPILOGUE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-bwd-epilogue-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_BWD_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-bwd-fuse-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_BWD_NORM1_RESID` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-norm1-resid-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_BWD_NOSYNC` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-bwd-nosync-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_HEAD_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-head-fuse-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_NOSYNC` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-nosync-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_PARAM_VIEWS` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-param-views-train, afn-lm-param-views-forward | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LM_WGRAD_SPLIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-wgrad-split-train | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_LOSS_FUSED` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-lossfused-mlp, afn-optim-lossfused-samba | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA1_CHUNKSCAN` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-chunkscan | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA1_FUSE_IN` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-fusein | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA2_SSD_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m2-ssdmma | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA3_BWD_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-m3-bwd-arena-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA3_BWD_CHUNK` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-m3-bwd-chunk-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA3_SISO_FUSED` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m3-sisofused | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-all, afn-mamba-m2-all, afn-mamba-m3-all | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-arena, afn-mamba-m2-arena, afn-mamba-m3-arena | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA_DEVICE_REFUSAL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-devrefusal, afn-mamba-m2-devrefusal, afn-mamba-m3-devrefusal | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA_PROJ_EPILOGUE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-m1-proj, afn-w2-epi-m2-proj, afn-w2-epi-m3-proj (+2) | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MAMBA_PROJ_EPILOGUE + AFN_MAMBA_PROJ_SPLITK` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-splitk-fwd, afn-w2-epi-samba-splitk-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MLP_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-all | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MLP_FUSED_STEP` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-fused | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MLP_MULTISTEP` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-multistep | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_MLP_RESIDENT` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-resident | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPTIM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-all-mlp, afn-optim-all-samba, afn-optim-all-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPT_CLIP_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-clipfuse-mlp, afn-optim-clipfuse-samba, afn-optim-clipfuse-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPT_FUSE_SCAN` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-fusescan-mlp, afn-optim-fusescan-samba, afn-optim-fusescan-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPT_MULTITENSOR` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-multitensor-mlp, afn-optim-multitensor-samba, afn-optim-multitensor-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPT_RESIDENT_STATE` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-resident-mlp, afn-optim-resident-samba, afn-optim-resident-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_OPT_VEC4` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-vec4-mlp, afn-optim-vec4-samba, afn-optim-vec4-lm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-all-training-fwd, afn-samba-all-training-step, afn-samba-all-mamba-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-arena-fwd, afn-samba-arena-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_DEVICE_ADMIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-admit-fwd, afn-samba-admit-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_EMB_ATOMIC` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-emb-atomic-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-fuse-fwd, afn-samba-fuse-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_SAMBA_FUSE + AFN_SAMBA_HEAD_GEMM` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-head-fwd, afn-w2-epi-samba-head-step | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AF_FAST_NOFILL` | adafactor / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-nofill-adafactor | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AF_FAST_RESIDENT` | adafactor / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-resident-adafactor | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `BPE_ALL` | bpe-encode / enwik8; bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-all, bpe-encode-all | - | OPEN | neural lines run after the classical queue |
| `BPE_ENCODE_DEVICE` | bpe-encode / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-encode-dev | - | OPEN | neural lines run after the classical queue |
| `BPE_ENCODE_DEVICE + BPE_LIVEBUF` | bpe-encode / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-encode-livebuf | - | OPEN | neural lines run after the classical queue |
| `BPE_GROUP_FILTER + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-filter | - | OPEN | neural lines run after the classical queue |
| `BPE_LIVEBUF + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-livebuf | - | OPEN | neural lines run after the classical queue |
| `BPE_MERGE_BATCH + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-batch | - | OPEN | neural lines run after the classical queue |
| `BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-dev | - | OPEN | neural lines run after the classical queue |
| `LN_FAST_NOFILL` | layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-nofill-layernorm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `MOE_FAST_MMA (+ _KB32, _WIDE, _PF)` | moe / synthetic | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-moe-* | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `OPT_FAST_MAP_DOWN` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-mapdown-adagrad, gapoptim-mapdown-rmsprop, gapoptim-mapdown-adamax, gapoptim-mapdown-nadam | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `OPT_FAST_PIPE_CH` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-ch2m-adagrad, gapoptim-ch2m-rmsprop, gapoptim-ch2m-adamax, gapoptim-ch2m-nadam | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `OPT_FAST_RAW_DOWN` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-rawdown-adagrad, gapoptim-rawdown-rmsprop, gapoptim-rawdown-adamax, gapoptim-rawdown-nadam | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SCHED_FAST_INLINE` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-inline-lrexp | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SCHED_FAST_INLINE,SCHED_FAST_P64` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-inlinep64-lrexp, gapoptim-schedcheck | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SCHED_FAST_P64` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-p64-lrexp | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_LSTM_SCAN` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-scan-regsyn, gaplstm-scan-regtaxi, gaplstm-scan-clftaxi, gaplstm-scan-clfsyn | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-smem-regsyn, gaplstm-smem-regtaxi, gaplstm-smem-clftaxi, gaplstm-smem-clfsyn | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM + SEQ_FAST_LSTM_WGRAD` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-all-regsyn, gaplstm-all-regtaxi, gaplstm-all-clftaxi, gaplstm-all-clfsyn | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_LSTM_WGRAD` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-wgrad-regsyn, gaplstm-wgrad-regtaxi, gaplstm-wgrad-clftaxi, gaplstm-wgrad-clfsyn | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_MAP_DOWN` | adafactor / synthetic; layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-mapdown-adafactor, gapoptim-mapdown-layernorm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_PIPE_CH` | layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-ch2m-layernorm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `SEQ_FAST_RAW_DOWN` | adafactor / synthetic; layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-rawdown-adafactor, gapoptim-rawdown-layernorm | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run |
| `AFN_GEMM_EPILOGUE` | neural | lane/apple-fast-neural @ 600237d7c | - | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run; the fused bias/activation/residual GEMM epilogue |
| `AFN_GEMM_KB` | neural | lane/apple-fast-neural @ 600237d7c | - | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run (tunable; the K block, default 32) |
| `AFN_GEMM_CORES` | neural | lane/apple-fast-neural @ 600237d7c | - | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run (tunable; the split-K core count, default 80) |
| `AFN_GEMM2_SWZ_G` | neural | lane/apple-fast-neural @ 600237d7c | - | - | MERGED-UNMEASURED opt-in (neural; Andrew Oct 3) | on main via lane/neural-merge-unmeasured, default OFF; no A/B run (tunable; the swizzle group under AFN_GEMM2_SWIZZLE) |

## Gap kapprox2 (lane/apple-fast-gap-kapprox2, Oct 3)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `KSHAP_FAST_BATCH` (`_OFF`) | kernel-shap / istella | lane/apple-fast-gap-kapprox2-kshap @ d95ede061 | kap2-kshap-batch-istella | 27011 -> 15325 | KEPT | regression from 57a297161 (one row per chunk: fresh 180 MB device buffer + ~3 GB/s download into a fresh host array, 2(d-1) pivot/elim launches per row); pooled device synthetic buffer, reused host buffer, one solve sweep per ~74 rows; rel_error_vs_exact 4.378e-09 both arms; also on PermutationExplainer (buffer reuse) |
| `KSHAP_FAST_BATCH` (`_OFF`) | permutation-shap / istella | lane/apple-fast-gap-kapprox2-kshap @ d95ede061 | kap2-pshap-batch-istella | 34198 -> 28236 | KEPT | pooled device synthetic buffer + reused host buffer (388 MB/row); rel error identical; still ~1.6x the 0.8.34 17.2 s |
| `KM_FAST_PTR_IN` (`_OFF`) | rbf-sampler / istella | lane/apple-fast-gap-kapprox2-kmeth @ 4624856ca | kap2-km-ptrin-rbf-istella | 121 -> 94 | KEPT | regression 85.7 -> 120 from 768c358e5 / c-core (multi-thread copies removed); transforms upload X from the caller's pointer, finiteness scanned on the device; kernel_rel_error identical .142 |
| `KERNEL_FAST_NYS_RR_EIGH` (`_OFF`) | nystroem / taxi | lane/apple-fast-gap-kapprox2-kmeth @ 4624856ca | kap2-km-nysrr-taxi | 533 -> 190 | KEPT | round-robin parallel Jacobi instead of the one-block serial eigh of the 256 x 256 basis kernel; kernel_rel_error .04561 -> .04503 |
| `SPLINE_FAST_FUSED` (`_OFF`) | spline / istella, taxi | lane/apple-fast-gap-kapprox2-spline | kap2-spl-fused-{istella,taxi} | 48.1 -> 11.6, 34.2 -> 10.8 | KEPT | fit allocates no n*d arena block when nothing sorts (main's came back from the device unread, 64 MB) and takes count/min/max from the blocked units; fit_transform one x_prep program; output digests identical |
| `SVGP_FAST_SYMTILE` (`_OFF`) | svgp / taxi | lane/apple-fast-gap-kapprox2-svgp | kap2-svgp-symtile-taxi | 611 -> 463 | KEPT | B = Kuf Kfu over the upper 4x4 blocks only, mirror written (two_prod commutes); r2/rmse identical |
| `SVGP_FAST_COLSPLIT` (`_OFF`) | svgp / taxi | lane/apple-fast-gap-kapprox2-svgp | kap2-svgp-colsplit-taxi | 613 -> 438 | KEPT | column phase's 4 triangular solves as 4m items, S/C/Sigma^-1 as m^2 items (was 512 serial threads); r2/rmse identical |
| `ACHI2_FAST_DEVCHECK` | additive-chi2 / istella, taxi | lane/apple-fast-gap-kapprox2 @ 22aaa623e | kap2-achi2-devcheck-{istella,taxi} | +1%, 0.9 -> 2.8 | DROPPED | device flag for the X < 0 check costs a launch + sync more than the host min pass at these sizes; not merged |
| `XD_FAST_GRP_FUSED` (`_OFF`) | gaussian-rp, sparse-rp / taxi, istella | lane/apple-fast-gap-kapprox2-grp | kap2-grp-fused-{taxi,istella}, kap2-srp-fused-{taxi,istella} | grp taxi 3.4 -> 2.1, istella -4.6%; srp taxi -10.7%, istella -8.5% | KEPT | fit in one binding call and one synchronize (pooled scan partials, matrix drawn+scaled/selected by one kernel, both downloads in one queue); distortion identical. sparse-rp istella distortion gap vs sklearn: no bug (density, scaling, laws = sklearn's; one 10-component draw) |
| `KSHAP_FAST_SIGNGRAM` | kernel-shap / istella | lane/apple-fast-gap-kapprox2 @ 4fd464a43 | kap2-kshap-sign-istella (A = BATCH) | -1.4% | DROPPED | noise; sign adds instead of soft-f64 products in the normal equations (same words); not merged |

## Fixes without a switch (merged; not experiments)

| change | branch @ sha | main | note |
|---|---|---|---|
| LinearRegression TSQR fix (launch floor + power-of-two equilibration) | lane/apple-fast-tsqr @ 7de01a481 | 83390b7ca | ols istella 65,858 ms r2 .157 -> 2,935 ms r2 .3325; ID check owed (host lacks lm_col_sums) |
| Prophet team fit from raw Args pointers | lane/apple-fast-prophetfix @ 2fcb29c68 | (merged, see LEDGER) | forecast_rmse restored 1.0152 / 32.034 |
| Optimizer resident handle (Python only) | lane/apple-fast-optfix @ ea3c917de | (merged, see LEDGER) | adamax/adagrad/rmsprop/nadam/lion/lamb ran again |
| MoE shape check (Python only) | lane/apple-fast-moefix @ baa5d967a | (merged, see LEDGER) | moe synthetic 788.7 ms |
| RR_EIGH scope to LDA/QDA | lane/apple-fast-eighscope @ 73f3a856d | 7edf6d895 | iterative-imputer istella 8,805 -> 4,596 ms |
| purity: host steps out of LP/LS/PageRank stop sums, SVGP ELBO folds | lane/apple-fast-purity | a5e98534a | IDENTICAL bits change on every vendor + host |
| py2mojo-core / -prep / -decomp / -neighbors / -linear: Python loops into Mojo | lane/apple-fast-py2mojo-* | 93d3522cd, 5ebff5c29, 05fffc97d, 12f304993, f72fbf64a | no switches |

## GPU purity 2 (lane/apple-fast-purity2, Oct 3)

Fixes of UNOWNED rows of `tools/hooks/host_routes_baseline.tsv`. Arm A of each A/B is the `_OFF` define (the old cost class), arm B the fix (default).

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `PURITY2_1` (`_OFF`) | gmm / istella | lane/apple-fast-purity2 @ 6f1ad9cf7 | purity2-1-gmm-istella | owed | OPEN | FAST mean log-likelihood: up to 64 partial blocks + a fold, not one 1024-thread block over n (mixture/checks/estep.mojo) |
| `PURITY2_2` (`_OFF`) | logreg / istella, taxi | lane/apple-fast-purity2 @ 6f1ad9cf7 | purity2-2-logreg-{istella,taxi} | owed | OPEN | QN loss sum + bias mean in QN_TILED's tile order (two passes), not one block over n (glm/impl/qn/glm_base.mojo); IDENTICAL softmax loss word changes on every vendor + host |
| no switch | rank-one Cholesky update (LARS) | lane/apple-fast-purity2 @ 6f1ad9cf7 | none | - | merged | the dot in the blocked-then-tree order (cholesky/logdet_fold.mojo sqsum); IDENTICAL bits change for m > 256, host oracle the same |
| no switch | ridge (svdEig) | lane/apple-fast-purity2 @ 6f1ad9cf7 | none | - | merged | descending eigen order ranked on the device (ties to the lower index), host column the same; bits change only on exact eigenvalue ties |
| no switch | permutation test | lane/apple-fast-purity2 @ 6f1ad9cf7 | none | - | merged | observed statistic on the device in host_tree_sum's order; no bit change |
| no switch | HDBSCAN / single linkage FAST (Boruvka, m > 4096) | lane/apple-fast-purity2 @ 6f1ad9cf7 | purity2-boruvka-check | - | merged | every Boruvka round on the device (atomic-min edge passes, scan, label propagation, radix sort); same MST by construction; correctness CMD queued |
| no switch | x_trees apply refusal, tsqr NaN refusal | lane/apple-fast-purity2 @ 6f1ad9cf7 | none | - | merged | error-path scans on the device (atomic min of the first bad key) |
| no switch | potrf panel / strip diag / sabotage arms, householder QR leaves | lane/apple-fast-purity2 @ 6f1ad9cf7 | none | - | merged | small-launch notes: the launches factor a w x w block (n is the row stride) or one TSQR leaf |
| no switch (quality) | FactorAnalysis (algos): two-pass column mean + cancellation-free psi update (q_j sum_i V_ij^2 w_i); both modes, all vendors | lane/apple-fast-quality-glmfa @ 2e8321b00 | qglm-factor-analysis-istella | istella mean ll 89.03 -> 99.49 (sklearn 98.1; IDENTICAL nv/amd before 77.80), 6172 -> 10351 ms (more EM iterations now that psi keeps converging; still ~0.3x sklearn); taxi -14.8237 unchanged | KEPT (quality first) | GLM rows (gamma/tweedie taxi, tweedie istella) judged no change: FAST = IDENTICAL to 1e-6 and sklearn equally negative (gamma taxi -232.96, tweedie taxi -10.07, tweedie istella -21.87 vs ours -24.41) |

## Gap linalg2, kernel-pca + incremental-pca (lane/apple-fast-gap-linalg2-kpca, Oct 3)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `KPCA_FAST_LANCZOS_DEV` | kernel-pca / taxi; kernel-pca / istella | lane/apple-fast-gap-linalg2-kpca @ 134dca742 | gl2k-kpca-lzdev-taxi, gl2k-kpca-lzdev-istella | kernel-pca taxi -9%; istella -12.5% | KEPT | M2 quality-only gl2k-kpca-quality-r3: eig rel diff 1.795e-07, transform angle 1.879e-06 rad; FAST+Apple default, -D MOJOLEARN_KPCA_FAST_LANCZOS_DEV_OFF reverts |
| `IPCA_FAST_DEV` | incremental-pca / taxi | lane/apple-fast-gap-linalg2-kpca @ 134dca742 | gl2k-ipca-dev-taxi | incremental-pca taxi 159 -> 136 | KEPT | -14.5%; output digest bit-identical; each batch stacked on the device, public arrays read once; FAST+Apple default, -D MOJOLEARN_IPCA_FAST_DEV_OFF reverts |

## Oct 4 manager takeover

FAST bit changes are allowed. Acceptance requires faster M3 timing and no quality decline beyond noise; bitwise identity is sufficient evidence in some cases, not a universal requirement. Near-noise single-run results remain inconclusive.

| Experiment | Branch / measured head | M3 tag | A → B (ms) | Quality | Verdict |
|---|---|---|---|---|---|
| SGDOC_FAST_PAR, Istella | lane/apple-fast-sgdoc-parallel @ 61710a5c8 | sgdocp-istella | 75537.985 → 181.313 | J .10000456 → .1; flagged .04042 → 0 | HOLD: anomaly behavior unresolved; old-base comparator |
| SGDOC_FAST_PAR, taxi | lane/apple-fast-sgdoc-parallel @ 61710a5c8 | sgdocp-taxi | 58334.248 → 16.521 | candidate J .1, flagged 0 | HOLD: anomaly behavior unresolved; current-main comparison owed |
| SGDOC_FAST_TAIL (`_LONG` = 4x tail), taxi + Istella | lane/apple-fast-w2-sgdoc | w2-sgdoc-q (quality), w2-sgdoc-tail-taxi / -istella | taxi 42142.4 -> 153.7 ms, Istella 75533.9 -> 407.0 ms; quality PASS | tools/sgdoc_tail_pair.py quality: A main vs B over 5 seeds on restandardized rows-small; ff, J, abs(w), decision spread inside main's seed range, flag Jaccard >= main's own cross-seed floor, shifted (uncentered) taxi B == A bit for bit | DEFAULT (FAST+Apple), rollback MOJOLEARN_SGDOC_FAST_TAIL_OFF; parallel algorithm still owed. Different from SGDOC_FAST_PAR: keeps SGD's anomaly behavior instead of the converged w = 0 (flags 0). With learning_rate='optimal', w_T = S_T / (alpha (t0 + T)); on centered data (S, R) is a stationary chain, so main's own per-sample kernels run only the last 65,536 steps of the real schedule (same epoch orders, same t) from w = 0, intercept = 1. Gate: one class, optimal, l2, fit_intercept, shuffle, tol=None, no sample weights, every column abs(mean) <= 1e-3 sd (two grid kernels), else main's full run. The tail is still a per-sample chain (main's kernel), 300x shorter on the board |
| ARIMA_FUSED_EVAL_TAIL | lane/apple-fast-arima-batched @ 1a627f709; kernel 217e821e5 | gap26-arima-tail-synthetic / taxi-hourly | synthetic 14498.225 → 13569.230 (-6.4%); taxi-hourly 24231.028 → 22705.348 (-6.3%) | Both digests and RMSE unchanged; gap26-arima-quality-fixed PASS, 44 byte-identical selected-order/parameter/likelihood/forecast arrays | KEEP: default/OFF promotion 311d5233e; taxi RMSE 74.6591 remains worse than opponent 68.21, quality gap under review |
| TARGET_SCRATCH (`_OFF` rollback) | lane/apple-fast-target-current @ 4d1ea20b2; promotion on main12fdd6697 | gap26-target-current-taxi / istella | taxi 270.947625 → 203.743667 (-24.8%); istella 250.423916 → 179.818916 (-28.2%) | Same per-dataset digests; gap26-target-current-quality PASS108 exact arrays plus independent smoothing oracle | KEEP for promotion: recorded maintenance windows clear; one pair per arm, variance unestimated; default/OFF compilation owed, not merged |
| EIGH_FAST_TANGENT | lane/apple-fast-linalg-20261004 | gap26-eigh-tangent (proposed) | Pending | residual/eigenvalue error/orthogonality required | OPEN; first compile alias error fixed, rebuild pending |
| MCD_BATCH_COMPAT | lane/apple-fast-mcd-exact | gap26-mcdcompat-taxi (proposed) | Pending | fitted covariance, precision, support, Mahalanobis distances, predictions | OPEN; compilation pending |
| ARIMA_EXACT_STEADY | lane/apple-fast-arima-batched @ a670ce8c2 (removed in 217e821e5) | NOT RUN | No new timing | Equivalent to earlier failed P_FIX covariance fixed-point experiment | ABANDONED before M3; prior P_FIX +165%, combined ASYNC +22–64%; avoids duplicate experiment |
| SYM_CTR_PERM_BATCH | lane/apple-fast-batch @ 3150d75c1 | sym-ctr-perm-batch-taxicat-x | 27161 → 27196 (+0.1%) | AUC .630994 → .631048, logloss .528561 → .528534 | DROP-speed: no gain on old base |
| CTR_PREP_SHARED | lane/apple-fast-batch @ 3150d75c1 | sym-ctr-prep-shared-taxicat-x | 27548 → 26357 (-4.3%) | AUC .631249 → .630964; noise not established | HOLD: old base; quality decline requires assessment before any current-main verification |
| CTR_INDEX_FUSED | lane/apple-fast-batch @ 3150d75c1 | sym-ctr-index-fused-taxicat-x | 27240 → 25566 (-6.1%) | AUC .630808 → .630766; logloss .528548 → .528535 | HOLD: old base; candidate only, no merge without current-main A/B and quality |

Gap26 accounting evidence and converted arm-B records: [ab/gap26/README.md](ab/gap26/README.md). Label-direct experiment verdict is owned by its promotion branch; its two measured board cells and independent quality receipt are preserved in this accounting bundle.


## Label-direct promotion (2026-10-04)

| define | algorithm / dataset | measured branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `MOJOLEARN_LABEL_DIRECT` -> `MOJOLEARN_LABEL_DIRECT_OFF` | label-binarizer / taxi, istella | lane/apple-fast-label-direct @ dbe2ab85a (base35a72c5ac) | gap26-label-taxi, gap26-label-istella; quality gap26-label-quality | taxi348.254458 -> 327.625833; istella33.176500 -> 27.578792 | KEEP, default merged on main1975c2dc5 | One M3 run/arm, -5.9%/-16.9%, respective digests0703e6f6396640c4/de224838a841fa8c identical. Independent public-output oracle PASS both arms, 41,697,776 cells each, fitted classes/codes/inverse checked. Source review against main61ea51757: relevant drift comments only. Manager verified default/OFF builds rc0 before merge. |

Quality review: `tools/label_fast_quality.py` checks every LabelBinarizer
training/query indicator against an independent definition, actual fitted
classes, inverse-transform values and LabelEncoder codes. It also checks
MultiLabelBinarizer classes and train/query indicators, including empty rows,
unseen labels and duplicates. Covered numeric cases include257 classes,
sparse integer ranges, binary/single class, negative/range/fractional fallbacks,
signed zero, and non-default positive/negative output labels. Actual bounded
M3 log reports `LABEL-FAST-QUALITY status=PASS checked_cells=41697776 captures=84`
for both arms followed by `QUALITY-PAIR-PASS gap26-label`.

The wrapper did not request `--dump`, so it did not compare84 saved arrays
between arms; each arm independently passed the mathematical oracle. Board
quality itself is only output_shape and is insufficient alone; the oracle and
unchanged board digests provide the quality evidence. Coverage gaps: invalid
NaN/Inf labels, integers not exactly representable as FP32, noncontiguous/empty
input, string/bool fallbacks and MultiLabelBinarizer inverse are not explicitly
exercised. These remain on existing guarded fallback routes; no claim is made
that those cases were tested. Encoder/multilabel correctness was checked but
speed was measured only for LabelBinarizer; do not update their timing rows.

## MCD compatibility repair review (2026-10-04)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `MOJOLEARN_MCD_BATCH_COMPAT` | MinCovDet / taxi cap3000 | lane/apple-fast-mcd-exact @ ab4265c9a | gap26-mcdrepair-small-ready | quality-only fit5595.80 ->828.96; full-board timing gated | HOLD-quality | location_rel.032817, covariance_rel.085678, precision_rel.999817, distances_rel.997423; support Jaccard.941431, raw_support.798209, both raw ranks10. Collective-entry repair removed launch/convergence failure but fitted values remain different. Same flags(all true) do not establish quality. |

Audit/proposal in `ab/mcd-compat-review.md`: main uses native Apple MMA for
covariance, weighted eigenvector Gram and Mahalanobis products; COMPAT uses
scalar FMA chains. Main covariance also splits K for support>=1024, unlike
COMPAT's4096-term folds. This concrete arithmetic mismatch can amplify through
singular determinants and candidate selection, but saved final fits alone do
not identify the first divergent stage. No thresholds changed. No new numeric
candidate is approved or claimed fixed; full timing remains gated.

## MCD MMA repair candidate (2026-10-04)

| define | algorithm / dataset | branch / base | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `MOJOLEARN_MCD_BATCH_MMA` | MinCovDet / taxi, narrow d<=64 | lane/apple-fast-mcd-mma (base d4bb2b795) | gap26-mcd-mma-{small,ee-small,full,ee-full} | MCD taxi 70877.7 -> 3587.2 ms; EE taxi 70092.1 -> 3627.9 ms; all MCDQ-PAIR-PASS | DEFAULT (FAST+Apple), rollback `MOJOLEARN_MCD_BATCH_MMA_OFF` | Actual repair after scalar COMPAT quality failure: use main's existing MMA/split-K launcher per candidate for covariance, weighted Gram and Mahalanobis. Batched control/eigen/support work retained. No host model computation, no threshold changes. See ab/mcd-mma.md; compile, capped quality and conditional full timing owed. |
| `MOJOLEARN_MCD_BMMA` | MinCovDet, EllipticEnvelope / taxi | lane/apple-fast-w2-mcd2 (base b2b1c22bc) | w2-mcdb-t-mcd-taxi, w2-mcdb-t-ee-taxi; quality w2-mcdb-q-mcd-taxi, w2-mcdb-q-ee-taxi | M3 one run per arm: MCD taxi 3002.9 -> 294.3 ms; EE taxi 3010.3 -> 299.3 ms; both MCDQ-PAIR-PASS (1% fitted state, .99 support) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_MCD_BMMA_OFF` | One batched matrix-unit GEMM launch per product per step instead of 3 x nc launches; inactive candidates skipped. Same tile/K split per candidate as main. Hypothesis: ~310k phase-B launches dominate 3.6 s. See ab/mcd-next.md. |
| `MOJOLEARN_MCD_WIDE` | MinCovDet, EllipticEnvelope / istella (d220) | lane/apple-fast-w2-mcd2 (base b2b1c22bc) | w2-mcdw-q-mcd-istella-r1, w2-mcdw-q-ee-istella-r1 (quality); w2-mcdw-t-* (timing) | main fallback EE istella 1253937 (py2mojo-decomp-elliptic-envelope-istella arm A) -> B only: MCD 86333.5, EE 86640.8; both MCDQ-PAIR-PASS (istella cap 3000, B = BMMA + WIDE) | DEFAULT (FAST+Apple, needs MCD_BMMA), rollback `MOJOLEARN_MCD_WIDE_OFF` | Batched search for 64 < d <= 256: block-per-candidate LU, 256-entry pinvh tables. B-only timing gated on istella cap3000 quality PASS. ~20 GB peak at board size. |
## AutoARIMA order batching current-main integration (2026-10-04)

| Experiment | Branch / baseline | Evidence | Verdict / next step |
|---|---|---|---|
| `ARIMA_ORDER_BATCH`, original small quality | lane/apple-fast-arima-orders @ eae73e1f0; kernel457160e23 | Manager reports gap26-orders-quality PASS18 arrays on the 128-observation fixture; no new timing | Historical small-only evidence; not a full-quality or current-main acceptance |
| `ARIMA_ORDER_BATCH`, fused-tail integration | lane/apple-fast-arima-orders-current; merged main d4bb2b795 (includes accepted fused-tail311d5233e) | Both arms now use current-main fused tail in shared prepare/finish; source changed, stale arms invalid | OPEN, opt-in only. Rebuild arima A/B, full512/2048 quality before one M3 timing/arm on synthetic and taxi-hourly; no opponent reruns. [Plan and exact commands](ab/arima-orders-current.md) |

## M3 repaired checks — 2026-10-04 09:00 UTC

- `MOJOLEARN_CHOL_FAST_TRI_SYRK`, lane/apple-fast-chol-20261004 measured `ca5ea4b6e` (kernel `e724b7777`), `gap26-chol-fixed-synthetic`: A268.772791 -> B288.278250 ms (+7.3%), one run/arm. Digest `8818853dfae997da` and relative residual1.659095968e-7 identical. SPD/solve and failure-info quality checks pass. **DROPPED-slower**, no default or board change; comment retained beside candidate gate on its lane.
- `MOJOLEARN_EIGH_FAST_TANGENT`, lane/apple-fast-linalg-20261004 `131a0d78a`, `gap26-eigh-rayleigh-run-ready`: A43796.817 -> B51254.836 ms (+17.0%), one run/arm. Eigenvalue error6.08669e-5 ->3.52564e-7; relative residual5.37499e-5 ->4.82917e-6. Repaired quality fixture pair passes. **HOLD-speed**: accuracy improved but no speed win; keep opt-in, optimize coefficient reuse before new measurement. Existing opponent-quality board hold remains.
- `MOJOLEARN_MCD_BATCH_COMPAT`, lane/apple-fast-mcd-exact `ab4265c9a`, `gap26-mcdrepair-small-ready`: capped3000 taxi only, A5595.804 -> B828.962 ms. **HOLD-quality**: covariance relative difference.08568, precision.99982, support Jaccard.94143. No full-data timing or board update. Artifact review finds final covariance rank7 ->6 at pinvh cutoff despite both raw ranks10; matching all-true flags is insufficient. Failure and proposed MMA-compatible repair documented on review branch `lane/apple-fast-mcd-review` at `0b5b3db39`; do not merge rejected candidate ancestry into main.

Raw M3 receipt: `~/mojolearn-evidence/apple-fast/sync/quality-repairs-results-0900.txt`; individual tags under `~/mq/out/`. Source review for queued PT `bc112b172`: subsequent main x_prep changes only add label-specific binding/dispatch/Python paths; power-transformer implementation is unchanged.

- `MOJOLEARN_GBDT_DW_BRIDGE_SCAN`, measured kernel `2519f4867`, helper `f193454e7`, `gap26-dwcurrent-taxi`: A10279.095 -> B10827.749 ms (+5.34%), one M3 run/arm. AUC .632554 -> .632211; logloss .527920 -> .528002. **DROPPED-slower**; changed quality has no established noise bound, so no quality-equivalence claim. No default or board change. Receipt `~/mojolearn-evidence/apple-fast/sync/dwcurrent-result-0905.txt`; full M3 `~/afc-def/gap26-dwcurrent-taxi/`.

## PowerTransformer compensated score failure (2026-10-04)

| Experiment | Source / tag | Quality | Verdict / next step |
|---|---|---|---|
| `PT_SCORE` | lane/apple-fast-pt-precision @ bc112b172; gap26-pt-score-quality | Stress per-column worst regression: lambda .01330737 vs1e-5 tolerance; NLL/sample4.083e-7 vs1e-7; transform RMS9.606e-5 vs1e-5; normality nonfinite/shape failure. Box-Cox improves and passes | HOLD-quality; timing skipped, no speed claim/default. Diagnose saved stress columns and distinguish nonfinite reference from candidate before repair. Tolerances unchanged; source context [PT_SCORE.md](PT_SCORE.md) |

Saved-array diagnosis: PT stress columns0–5 improve to ~1.5–1.8e-7 transform RMS. Regressions are near-constant columns6/7: reference lambdas52.6079/-63.3852 exceed main and candidate[-8,8] interval; f32 per-row log/transform/derivative precision remains before compensated reduction. The normality NaN belongs to reference column7 (constant sklearn transform, std0), not GPU outputs. No threshold relaxation or speculative kernel repair; a stable shared score/output transform and corrected stable oracle are needed before retry. See [PT_SCORE.md](PT_SCORE.md) for exact per-column evidence.

### Measurement isolation audit — 2026-10-04 09:13 UTC

The M3 runner serializes jobs and A/B arms, but the manager ran filesystem scans
and cleanup outside the queue during this session. This violates full machine
isolation even though no simultaneous scored benchmarks were found. Recent
single-pair speed verdicts whose maintenance overlap cannot be excluded must be
treated as **HOLD-measurement**, superseding firm speed-only rejection claims
above (in particular the current depthwise tree comparison). Preserve raw times
and independent quality evidence; do not merge a candidate on an uncertain speed
result. No affected candidate was promoted from those recent rejected pairs.
Future heavy maintenance must share the serial queue or an explicit idle boundary.
The earlier ARIMA/label measurements predate this session's maintenance; this
audit alone does not establish interference with those measured promotions.

Detailed isolation audit and six identified scan-window tags: [MEASUREMENT_AUDIT_2026-10-04.md](MEASUREMENT_AUDIT_2026-10-04.md). Their speed verdict is HOLD-measurement, superseding historical speed-only verdicts. No overlap claim is made solely from a cleanup start timestamp.

## AutoARIMA order-batching default promotion prepared (2026-10-04)

| Experiment | Measured source / tags | A → B ms | Quality | Verdict / remaining gate |
|---|---|---|---|---|
| `ARIMA_ORDER_BATCH` → default + `ARIMA_ORDER_BATCH_OFF` | 7ba385b30; gap26-orders-current-synthetic / taxi-hourly | synthetic13780.316917 →9287.286750 (-32.6%); taxi-hourly22706.248042 →13747.215083 (-39.5%) | Fitted/order/likelihood/forecast full-quality PASS before timing; both board digests and RMSE identical (2.624119555 /74.659122441). Existing fused tail enabled in BOTH arms | KEEP, default merged on main dc2285bc0; manager default/OFF builds both rc0. Source review against main d1871643b preserved accepted fused tail. Taxi opponent-quality HOLD remains (74.6591 vs68.21). [Raw evidence and review](ab/arima-orders-default.md) |

## Wave 2 kernel features (lane/apple-fast-w2-kfeat, 2026-10-04, base 254e50a01)

Quality: `tools/kfeat_pair.py quality` (`tools/kfeat_quality.py`); tolerances in its docstring, fixed before any run.

| define | algorithm / dataset | before ms (board) | verdict | hypothesis / note |
|---|---|---|---|---|
| `KM_FAST_RBF_RESIDENT` | rbf-sampler / istella | 93.6 (sklearn 47.6) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_KM_FAST_RBF_RESIDENT_OFF`: M3 one run per arm istella 93.4 -> 76.3 ms; w2-kfeat-rbf-q-r1 PASS (byte-identical outputs) | fit_transform in one binding call and one wait: W, b stay on the device, X and the output in exact-size pooled buffers, no unused GEMM workspace, output on mapped memory (no zero pass); main's kernels, bit-identical. Main: ~10 waits, 5 fresh device allocations up to a few hundred MB. The 100 MB result download (~3 GB/s into host memory) stays |
| `XN_FAST_ACHI2_DEVSCAN` | additive-chi2 / istella, taxi | 12.3, 0.7 (sklearn 3.8, 0.4) | HOLD, opt-in: M3 one run per arm istella 11.6 -> 2.9 ms but taxi 0.9 -> 1.9 ms (w2-kfeat-achi2-*); needs a size gate | fit's negative check as a pooled upload + `negative_partial_kernel` + pinned partials, one wait (main: sequential host `X.min()`). Retry of the dropped `ACHI2_FAST_DEVCHECK`: that one allocated X's device buffer and the flag fresh per call; this allocates nothing after the first round. Refuses a negative behind a NaN (main's min can miss it) |
| `XN_FAST_SCHI2_MOJO_MT` | skewed-chi2 / istella, taxi | 7.3, 1.6 (sklearn 3.7, 0.6) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_XN_FAST_SCHI2_MOJO_MT_OFF`: M3 one run per arm istella 9.0 -> 2.2 ms, taxi 2.5 -> 1.5 ms; w2-kfeat-xn-q-r1 PASS (byte-identical outputs and refusals) | fit's legacy MT19937 draws (sklearn's numbers) in Mojo instead of a Python loop + nested list comprehensions, weights kernel and offsets in the same call, one wait; bit-identical (unlike the held `SCHI2_FAST_DEVRNG`, which changes the numbers) |
| `XD_FAST_SRP_STRAT` | sparse-rp / istella (quality HOLD), taxi | 15.2, 3.4 | DEFAULT (FAST+Apple), rollback `MOJOLEARN_XD_FAST_SRP_STRAT_OFF`: w2-kfeat-srp-q PASS, istella 40-seed mean distortion 0.946 -> 0.571, seed 7 1.883 -> 0.475 (sklearn 0.474); taxi mean 0.346 -> 0.233, seed 7 0.147 -> 0.264 (sklearn 0.381); speed istella 15.7 -> 15.9 ms (neutral) | quality fix, not speed: 1.883 vs sklearn 0.474 is the nonzero count of a dominant raw column (count 2 -> |2*1.483-1| = 1.97, count 1 -> 0.48; main's seed-7 matrix has 27 of 220 columns at count >= 2). Column-stratified systematic sampling: same per-entry rate, signs and scale, count = floor/ceil(k * density). Gate: 40-seed paired mean on the board blocks |

## Wave 3 kernel features (lane/apple-fast-w3-kfeat, 2026-10-04, OPEN)

Base 7cda81ab0 (lane/apple-fast-schi2-mt-default on main 5a5fc6395). Quality: `tools/kfeat_pair.py quality` (`tools/kfeat_quality.py`, fixture kfeat-v2: byte-identical arrays and refusals, tolerance zero, fixed before any run).

| define | algorithm / dataset | before ms (board) | verdict | hypothesis / note |
|---|---|---|---|---|
| `XN_FAST_ACHI2_DEVSCAN` + size gate | additive-chi2 / istella, taxi | 12.3, 0.7 (sklearn 3.8, 0.4) | DEFAULT (FAST+Apple), rollback MOJOLEARN_XN_FAST_ACHI2_DEVSCAN_OFF: M3 one run per arm (ea6b2035e) istella 11.2 -> 4.4 ms, taxi 0.9 -> 1.1 ms (below gate, main's host path: noise); w2-w3kf-xn-q PASS | device scan only from 2^22 entries (`XN_ACHI2_DEVSCAN_MIN`, x_neighbors/kfeat_dev.mojo); below it main's host `X.min()`. From w2's M3 points (taxi 1.1M entries host 0.9 / device 1.9 ms, istella 22.0M host 11.6 / device 2.9): host ~0.34 + 0.51 ms/M, device ~1.85 + 0.048 ms/M, crossing ~3.3M; the gate sits above it. Expected: istella ~2.9, taxi = main |
| `XN_FAST_SCHI2_LAZYW` | skewed-chi2 / taxi, istella | 1.6, 7.3 (sklearn 0.6, 3.7); M3 MOJO_MT taxi 1.5 | DEFAULT (FAST+Apple), rollback MOJOLEARN_XN_FAST_SCHI2_LAZYW_OFF (needs SCHI2_MOJO_MT): M3 one run per arm (ea6b2035e) taxi 1.8 -> 0.4 ms, istella 2.1 -> 0.9 ms; w2-w3kf-xn-q PASS | fit's remaining cost is one GPU round trip for 2,816 weights; fit now draws z and the offsets on the host (sklearn's sequential stream, as MOJO_MT) with no device work, and transform runs log, the pending weights kernel and the map in ONE call (main: 2 waits in transform + 1 in fit). Same kernels, same words |
| `KM_FAST_RBF_STAGED` | rbf-sampler / istella | 76.3 (sklearn 47.6) | DEFAULT (FAST+Apple), rollback MOJOLEARN_KM_FAST_RBF_STAGED_OFF: M3 one run per arm (ea6b2035e) istella 77.6 -> 57.1 ms; w2-w3kf-rbf-q PASS byte-identical | the 102 MB projection comes down through core/staged_download (pinned 8 MiB chunks, copy-out overlapped) instead of one raw host-pointer copy (~3 GB/s); the same transport took x_prep's label-binarizer taxi 563 -> 345 ms. Handing out the pinned memory itself is not done: the caller would read write-combined memory |

## SVGP wave 2 candidates (lane apple-fast-w2-svgp, 2026-10-04, OPEN)

Base b2b1c22bc. Rows: svgp istella 661 vs gpytorch-cpu 307 ms, taxi 437 vs 248 ms. Binding x_neighbors (`x_neighbors/iter_device.mojo`). FAST + Apple only; RBFTILE and BLKCHOL are default (rollbacks `MOJOLEARN_SVGP_FAST_RBFTILE_OFF`, `MOJOLEARN_SVGP_FAST_BLKCHOL_OFF`), BSPLIT is off unless defined. Quality gate `tools/svgp_fast_quality.py` (r2, rmse, elbo one-sided 1e-4, fixed before results); quality-gated pair `tools/svgp_fast_pair.py`.

| define | hypothesis | bits | verdict |
|---|---|---|---|
| `SVGP_FAST_BLKCHOL` | 3 float-float Cholesky factors (m = 512) took 1,024 dependent column launches; one launch per 16-column panel (64 launches), diag block factored in threadgroup memory, each entry the same chain as `_chol_col` | same chains (FAST contraction only) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_SVGP_FAST_BLKCHOL_OFF`: M3 one run per arm taxi 287.0 -> 212.5 ms, istella 336.6 -> 265.7 ms (measured without RBFTILE; combined retime owed); w2-svgp-blkchol-q PASS |
| `SVGP_FAST_RBFTILE` | scaled rbf (Kuu, Kfu, Ksu) was one thread per cell reading 2 d floats from global (istella d ~220: 51M cells); 64 x 64 block tiles, 16 features staged in threadgroup memory, 4 x 4 cells per thread, variance scale fused (no kbuf pass) | same per-cell fold (FAST contraction only) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_SVGP_FAST_RBFTILE_OFF`: M3 one run per arm istella 336.3 -> 299.1 ms, taxi 285.9 -> 283.9 ms; w2-svgp-rbftile-q SVGP-FAST-PAIR PASS (r2/rmse/elbo; A r2 taxi -0.19498, istella -0.10602) |
| `SVGP_FAST_BSPLIT` | SYMTILE's B launch keeps only 8,256 threads busy (32,768 rows deep each) and b = Kuf y only m threads; 4 (B) / 32 (b) row-slice float-float partials, summed slice-ascending | changes (float-float re-association, ~1e-14 rel) | OPEN, A/B owed |
## MiniBatchKMeans W2 residuals (lane/apple-fast-w2-clres, 2026-10-04)

Binding x_cluster (`x_cluster/minibatch_fast.mojo`), FAST + Apple only. Quality pair `tools/w2_clres_quality.py`.

| define | algorithm / dataset | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP` | MiniBatchKMeans / istella, taxi | w2-mbk-sumcmp-q | M3 one run per arm: istella 147.2 -> 144.7 ms; taxi 43.9 -> 39.9 ms; exact (centers, counts, labels, inertia identical) | DEFAULT (FAST+Apple), rollback `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_SUMCMP_OFF` | Sum kernel compacts its center's rows per 256-row chunk (prefix scan) and sums only those, same ascending order; same bits as main. |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG` | MiniBatchKMeans / istella, taxi | (not promoted) | n/a here | OPEN, opt-in | Last labelling pass as the CLS3_ROWGRP 32-thread-per-row assignment; reorders distance sums (labrg tolerance mode). |

## Manager verdicts, 2026-10-04 session 2 (rejected or held; candidates stay on their branches)

| define | rows | branch / source | evidence | result | status |
|---|---|---|---|---|---|
| `MOJOLEARN_EIGH_TANGENT_CACHE` | eigh synthetic | lane/apple-fast-eigh-cache 14764dbb8 | gap26-eigh-cache-synthetic-ready | A 43721.3 -> B 44070.7 ms; quality pair PASS (B eigenvalue error 3.5e-7) | DROP-speed, opt-in only |
| `MOJOLEARN_CAGRA_FAST_IVFG_LOWD` | cagra taxi | lane/apple-fast-w2-cagra 5d7d79cb5 | w2-cagra-lowd-q | taxi recall@10 A 0.997925 -> B 0.997125 (gate: B >= A); istella identical | DROP-quality; LOWD_SEEDS4 queued |
| py2mojo decomp default | elliptic-envelope istella | lane/apple-fast-py2mojo-decomp 9a550d46c | py2mojo-decomp-elliptic-envelope-istella | A 1253937 -> B 1273419 ms (first valid timings for this row) | no gain |
| `MOJOLEARN_SHAP_FAST_PIPE` | permutation-shap / kernel-shap istella | lane/apple-fast-w2-shap abd933572 | w2-shap-pipe-*-r1 | quality PASS (phi byte-identical); pshap 28217.4 -> 28271.8 ms, kshap 15418.1 -> 15446.0 ms | DROP-speed (no overlap gained), opt-in only |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG` | minibatch-kmeans | lane/apple-fast-w2-clres 4d80737b1 | w2-mbk-labrg-* | quality PASS; istella 146.6 -> 144.0, taxi 45.3 -> 46.3 ms | DROP-speed (noise), opt-in only |
| `MOJOLEARN_ARIMA_FIT_GROUPS` | autoarima | lane/apple-fast-w2-ts 596d0abbb | w2-ts-fitgroups-*-r2 | quality PASS bit-exact; synthetic 9225.3 -> 9268.2, taxi-hourly 13749.1 -> 13072.9 ms; diag: search 8.2 s of 13.7 s; 200-iter search RMSE 75.71 (worse than 74.66) | HOLD (no gain on eligible synthetic row) |
| PT_SCORE_STABLE centered | power-transformer | lane/apple-fast-pt-precision f88ed2cf6 | w2-pt-centered-quality | arm A dump failed: sklearn reference lambda col 7 not a local f64 NLL minimum (new oracle) | superseded: oracle v2 row below (DEFAULT) |
| PT_SCORE_STABLE centered (oracle v2) | power-transformer | lane/apple-fast-pt-precision 9c458698d | w2-pt-centered2-quality | quality PASS vs f64 centered-MLE reference (tolerances unchanged); taxi 293.4 -> 190.8 ms, istella 2206.6 -> 1530.5 ms | DEFAULT (FAST+Apple), rollback MOJOLEARN_PT_SCORE_STABLE_OFF |

## PT centered-score WIP checkpoint (2026-10-04)

`MOJOLEARN_PT_SCORE_STABLE`, lane/apple-fast-pt-precision, merged baseline12fdd6697:
uncompiled/unvalidated repair of heldbc112b172. Centered affine-equivalent
score/standardized output/inverse, span-derived bracket, stable same-lambda
float64 oracle with independent Decimal check; existing thresholds unchanged.
No builds/quality/timings yet; checkpointed for manager handoff, **not accepted**.
Update: oracle v2 (9c458698d) w2-pt-centered2-quality PASS; promoted to
FAST+Apple DEFAULT, rollback `MOJOLEARN_PT_SCORE_STABLE_OFF`.
See [PT_SCORE_STABLE.md](PT_SCORE_STABLE.md) for code scope and exact owed checks.
| `MOJOLEARN_GBDT_DW_FLAT_GRID` | gbdt-depthwise taxi | lane/apple-fast-w3-dw 1fb97706a | w2-w3dw-* | quality PASS (AUC +0.000107); 11054.9 -> 11822.4 ms | DROP-speed, opt-in only |
| `MOJOLEARN_ARIMA_SLAB` | autoarima | lane/apple-fast-w3-arima 40fedfcea | w2-w3arima-slab-* | quality PASS bit-exact; synthetic 9219.3 -> 9096.6, taxi-hourly 13763.6 -> 13519.0 ms | HOLD (gain within noise) |
| `MOJOLEARN_X_PREP_PINNED_OUT` | label-binarizer / multilabel-binarizer / target-encoder taxi | lane/apple-fast-w3-prep 92e4661e6 | w2-pinned-*-r1 | outputs sha-identical; timed call lb taxi 270.2 -> 64.5, mlb taxi 169.7 -> 87.7, te 201.6 -> 201.6 ms; BUT caller first read of the returned pinned array: lb taxi shape A 331.5 ms vs B 2153.2 ms, mlb 256.0 vs 1680.7 ms (call+read A ~604 ms vs B ~2199 ms) | DROP: moves the cost to the caller's read (write-combined pinned memory); benchmark-only gain |
| `MOJOLEARN_LU_FAST_TSLU` | lu-factor / lu-solve | lane/apple-fast-w4-tslu 5867b9fbe | w2-tslu-quality | gate PASS but hard matrices worse: plain1000 factor 2.54e-6 -> 3.12e-6, solve 1.20e-3 -> 1.66e-3; plain2051 factor 5.36e-6 -> 6.27e-6, solve 1.81e-3 -> 2.51e-3; board identical (no pivoting) | DROP-quality (tournament pivoting less stable than partial pivoting); not promoted regardless of speed |
| `MOJOLEARN_SEQ_FAST_VAR_FUSED` | var | lane/apple-fast-w4-small 74d233862 | (none) | M2 compile error: kernel signature not accepted by enqueue_function (sequence/var_fused.mojo:255); also a one-block launch | PARKED: conflicts with the parallel-GPU rule; needs a multi-block design |

| `MOJOLEARN_PREP3_MAXABS_POOL` / rollback `MOJOLEARN_PREP3_MAXABS_POOL_OFF` | maxabs-scaler istella | measured lane/apple-fast-w4-small@74d233862d74e2c8e48b7db38a24ec85a22b80a1; promotion lane/apple-fast-maxabs-pool-default based on main cd8d095cf | w2-w4s-maxabs-q; w2-w4s-maxabs-istella | M3 104.6 -> 17.0 ms; quality PASS 12/12 byte-identical arrays, exact NumPy max, pool reuse covered | DEFAULT; M2 default/_OFF both rc=0, merged. Only MaxAbs paths extracted; resample, VAR, KNN candidates excluded. Source review: main changes do not alter measured MaxAbs kernels, binding, pool, or Python route. Same synchronized caller-owned output; no deferred read. Evidence M3 `~/mq/out/w2-w4s-maxabs-q-quality/PASS.json`, `~/mq/out/w2-w4s-maxabs-istella.log`. No new timing or unmeasured board-cell claim. |

- MaxAbs promotion `033fbbe10`: M2 default and `MOJOLEARN_PREP3_MAXABS_POOL_OFF` compiled successfully (both rc=0), manifest `/Users/ec2-user/m2-arms/033fbbe1063bc986c4844e0119dd75fea3a9e4e6/x_prep/manifest.json`; integration merge records both parents.
| `MOJOLEARN_KPCA_RESIDENT` (rollback `MOJOLEARN_KPCA_RESIDENT_OFF`) | kernel-pca / taxi, istella; RBF, auto top-k (n > 200, 1–9 components) | measured lane/apple-fast-w4-decomp @ 34b4f6c72b489c16f8333ef1416ac23fd143cd59; promotion lane/apple-fast-kpca-resident-default from main cd8d095cfe4453ed808f37ca28247a999af1dd1a | w2-w4d-kpca-q; w2-w4d-kpca-taxi; w2-w4d-kpca-istella | taxi 850.7 -> 127.5 ms; istella 919.5 -> 201.0 ms | DEFAULT; M2 default + OFF both rc=0, merged bc61f3560; Python import PASS | Quality fixture w4q-v1 PASS: eigenvalue relative difference istella 9.8731e-7 / taxi 4.3144e-7 (<1e-4); largest subspace angle 3.7884e-6 / 1.7495e-6 rad (<1e-3). Quality call+first-read istella 1157.7+0.1 -> 359.9+0.1 ms, taxi 742.4+0.1 -> 126.1+0.1 ms. Returned arrays retain the same eager host storage: only intermediate kernel matrices stay resident. Source review: resident kernel construction, x_decomp primitives and Lanczos are unchanged between measured source and promotion base; lane-only PCA/LLE/RSVD changes excluded. Parallel device sqdist, elementwise and reductions; Python dispatch only. Other kernels/dense routes unchanged. Evidence M3 ~/mq/out/w2-w4d-kpca-q-quality/{A.log,B.log,compare.log,PASS.json}, timing tags above. |

| `MOJOLEARN_CHOL_FAST_NB512` | cholesky synthetic | lane/apple-fast-w4-linalg 19bb1c7a1 | w2-cholnb512-quality; w2-cholnb512-synthetic | quality PASS; 260.4 -> 265.7 ms | HOLD-speed: one sample per arm shows no benefit; opt-in only |
| `MOJOLEARN_RESAMPLE_FAST_ROW_GATHER` | resample taxi / istella | lane/apple-fast-w4-small 74d233862 | w2-w4s-rs-* | quality PASS; taxi 59.9 -> 53.3 ms; istella 386.5 -> 395.4 ms | HOLD: host-side row gather conflicts with GPU-only rule; no istella gain |
| `MOJOLEARN_XN_FAST_NAN_FIT_LEAN` | knn-imputer taxi | lane/apple-fast-w4-small 74d233862 | w2-w4s-knn-* | quality PASS; 2.2 -> 1.3 ms | HOLD: host reduction over downloaded counts; repair candidate f9c5887a8 uses GPU integer reduction, old timing does not validate it |
| `MOJOLEARN_MCD_SKIP_PINVH` | min-cov-det / elliptic-envelope istella | lane/apple-fast-w4-mcd d0b30bfe9 | w2-w4mcd-q/t-* | MCD 86853.6 -> 31891.1 ms; EE 86637.0 -> 31780.9 ms; small quality PASS, masks unchanged but final covariance/precision differ | HOLD pending saved-output oracle: identical raw outputs and masks feed unchanged split-K atomic covariance, which can explain drift but does not establish noise or no-regression |
| `MOJOLEARN_MCD_SKIP_PINVH` + `MOJOLEARN_MCD_DEFLATE` | elliptic-envelope istella | lane/apple-fast-w4-mcd d0b30bfe9 | w2-w4mcd-q-ee-istella-defl | gate PASS but flags Jaccard 0.9988687783; fraction 0.2946667 -> 0.2943333; precision relative delta 0.001521 | HOLD-quality: changed anomaly behavior requires evidence, not merely a permissive passing gate |

| `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS` / rollback `MOJOLEARN_EIGH_FAST_TRIDIAG_PANELS_OFF` | eigh synthetic, n>=512 with refusal fallback | measured e79a96e03d0331450a65544a7f55d06c7b140576; isolated promotion from main fc36f878c | w2-eigh-panels-q-20261004; w2-eigh-panels-t-20261004 | M3 43765.6 -> 700.3 ms; original no-regression criterion PASS all 8 fixtures, zero allowance; stricter absolute eig/Gram gates still FAIL and opponent-quality HOLD preserved | DEFAULT under original main-relative criterion; M2 default/OFF rc0 at 10cedc640, merged eb97b0948; absolute and opponent-quality holds retained. Panel factors now multi-block; previous single-block implementation excluded. Source review: two production files exactly measured except default switch/comment; intervening main changes do not affect x_decomp numerical path. |

| `MOJOLEARN_LU_FAST_MMA_DBUF` | lu-factor / lu-solve n8192 | lane/apple-fast-lu-mma-dbuf@5d4e5d5d5f61c93d49c03e7eb826130ca088dd9d | w2-lu-dbuf-q-20261004; w2-lu-dbuf-t-20261004 | quality PASS: exact LU/pivots/solutions and no-worse residuals on 8 fixtures; factor call+read 712.147958 -> 690.727875 ms; solve 778.036750 -> 778.825792 ms | HOLD-speed: factor gain about 3%, no solve gain; no default or board change. One scored run per arm, no replay. Source review identifies pivot-reduction synchronization as next distinct lever. |
| Rank-one Gaussian associative scan reference | AutoARIMA design only | lane/apple-fast-arima-scan-oracle@45b61672db3075e106c27917d971da0641ecbe2b | arima-assoc-oracle-v1 | 68 f64 algebra fixtures pass; 47 f32 checks and 9 finite-difference gradient checks fail | HOLD: rounded-Q model and derivative differences; not actual-main GPU evidence, no timings or promotion |

Saved-output MCD follow-up (`w2-w4mcd-oracle-{mcd-istella,ee-istella,mcd-taxi}-r1`,
analysis source `1388758c39afa56587b659d9ce2600d0fb0eb336`): HOLD remains.
Istella MCD precision error B 0.01008622 exceeds A 0.009929996 and prior A
0.009888035; train-distance error B 0.000575127 exceeds 0.000564590 and
0.000535111. EE distance/decision/offset errors also exceed both saved A
errors. Observed A–A variation does not establish an allowance for every
metric. Taxi final oracle errors improve/equal, but raw covariance is not
byte-identical and lacks an extra baseline. Reports preserve legacy capture
provenance limitations. Stabilizing split-K accumulation is a new candidate,
not retroactive validation of SKIP. DEFLATE remains HOLD for changed flags.

| `MOJOLEARN_DECOMP_FAST_MMA_K16` | standalone decomposition GEMM, non-split only | lane/apple-fast-decomp-mma-k16@d487c814fe59d35de111d392751af9e6ce06eb66 | w2-mma-k16-q-20261004; w2-mma-k16-t-20261004 | 12 oracle fixtures PASS; dense4096 call+read 85.504250 -> 91.724583 ms (resident 43.152292 -> 47.424042); update4096x256x4096 36.670541 -> 36.401583 (resident 30.276000 -> 30.508875) | HOLD-speed: dense slower, update essentially unchanged. No caller timings/default/board changes. Harness calls transpose1024 a changed shape, but dispatcher yields256 tiles and2 splits, so it is an UNCHANGED control, alongside Gram/thin controls. Its apparent host gain6.389875 ->3.939625 cannot be attributed to K16. One scored call per route/arm retained; no replay. Resident completion timing is a fence-inclusive observation, not pure GPU throughput. |

## W4 PCA pool isolated promotion (2026-10-04)

`MOJOLEARN_PCA_FAST_POOL_OFF`: FAST Apple DEFAULT, merged d8e7825cf; M2 default/OFF rc0 at promotion906fe59c9.
Measured source `34b4f6c72b489c16f8333ef1416ac23fd143cd59`, timing
`w2-w4d-pca-istella-r1`: M3 Istella 471.2 -> 217.8 ms. Quality
`w2-w4d-pca-q`: PASS 16 checks, 200,000 x 220 seeded ill-scaled input,
three consecutive fits to exercise dirty pool reuse, ten components;
means identical, eigenvalue relative differences about 1e-6, component
angles about 2e-6 to 3.5e-6 rad, reconstruction absolute differences
1e-12 to 1.6e-11. NaN refusal checked. Values are A/B differences, not
independent-reference accuracy claims. Manager source and noise evidence review completed.

This promotion is isolated on main `f690a6308`: estimators binding plus
PCA input pool/unused aliases only. No RSVD, LLE, KPCA or eigensolver changes.
The unchanged covariance MMA arm reads neither alias buffer; its input is
fully overwritten from the caller before reuse. Pool return follows final
synchronization and host output copies. Exceptions release owned buffers.
Pool keeps at most 2 GiB idle; board input occupies about 1.8 GB.
Outputs remain eager host arrays; no work is deferred to first read.
Unconditional getenv/perf_counter_ns calls and optional stage file logging
from the measurement source are removed entirely from the proposal.

Source comparison found no intervening change in this PCA numerical path;
main's newer x_decomp host eigh dispatch is a separate implementation.
FAST Apple and existing PCA_FAST_GRAM_MMA eligibility are retained; split-K
cases and IDENTICAL remain on fresh allocation. `_OFF` restores the original
allocation policy. M2 default/OFF builds passed;
no local compilation or timing was run. Broader shapes and modes were not
newly measured by this quality fixture.

Stored PCA reconstruction values (same X, same seed across all three fits):
A = [0.2964381628654726, 0.2964381628635881, 0.29643816285693114];
B = [0.296438162864011, 0.29643816286408387, 0.29643816286171426].
Lower is better. B fit1 and fit2 worsen by about 5e-13 and 4.8e-12,
within the observed A repeated-fit range of 8.54e-12. All three B values
also lie inside that A range. This supports noise-level differences on
this fixture only; three repetitions do not establish a general noise
bound. The unchanged split MMA atomic accumulation permits such variation.
Quality call times A [277.8, 248.0, 247.1] ms, B [264.4, 222.2, 221.9] ms;
all first reads round to 0.0 ms (not asserted literally zero).

Related candidates remain unpromoted: RSVD Istella 517.3 -> 501.3 ms and
taxi 200.8 -> 198.5 ms are under 5% at n=1, HOLD-speed/noise. LLE taxi
1783.8 -> 1039 ms is useful speed, but trustworthiness worsened from
0.7490329563692009 to 0.7490198410032471 (1.31154e-5); Istella improved
from 0.5841975142761169 to 0.584208390251185. LLE remains HOLD-quality:
an absolute-difference PASS alone cannot establish no regression and no
same-arm noise evidence is yet recorded. No LLE default proposal prepared.

Catalog G1/G5 standalone6abb76673, w2-catalog-g1g5-q-20261004-r2: HOLD for unrestricted use. Eleven cases byte-identical to incumbent; only NT vector79x1,K65 regresses scaled error7.05755e-9 ->1.78820e-8. Preserve existing GEMV fallback; shared adapter predeclares this exclusion and needs its own quality evidence. Callpath r2 failed import (missing PyInit export), not numerical quality; repair required.

| `MOJOLEARN_ARIMA_FAST_SCALAR_LL` | scalar Kalman likelihood, actual GPU kernels | lane/apple-fast-arima-scalar-k3-r2@6f09497abfe50dcda0b44ea7ee8c8e1540bd4aab | arima-k3-kernel-q-r2 | M2 A/B PASS; M3 kernel quality HOLD across13 groups, no scored timings | HOLD pending corrected actual-gradient check: all12 n>1 groups fail the harness gradient metric, but source review found its rounding tree differs from product ew_finish/ew_grad. This is not proof of product-gradient regression; separate innovation/likelihood failures remain. n=1 passes. Actual full-fit/forecast validation remains owed; no optimizer or timing admission. Scalar specialization is not approved merely by exact algebra. |

SHAP delta promotion preparation `e3264731af227951cb2ecfb80f81d359592486b5`
passed M2 default and `MOJOLEARN_PSHAP_DELTA_OFF` builds plus Python import
smoke. Measured20bbdc372 gives28222.8 ->14788.6ms and exact output words in
all3quality cases. Final default/rollback M3 quality pshap-delta-default-r1-quality-20261004 PASS: 13 arrays, A=default and B=OFF; exact source/binary hashes pinned. Approved for merge. Fresh source pin fixes only promotion arm ordering; no scored replay.

| `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE` / rollback `MOJOLEARN_LU_FAST_PIVOT_SHUFFLE_OFF` | lu-factor / lu-solve n8192 | measured6d82b61270dc5d7c65f1d3b847a91630c4e2bfcd; isolated default from0cbe0036a | w2-lu-pivot-shuffle-q-20261004; w2-lu-pivot-shuffle-t-20261004 | quality PASS10fixtures exact factors/pivots/solutions/info, no-worse residuals; call+read factor715.628750 ->653.702416ms, solve792.222333 ->733.531042ms | DEFAULT merged; promotion0769e1d92 M2default/OFF rc0. Existing parallel LUstep grid unchanged, exact directed reduction topology, same tie/NaN comparator; skips neutral initial-fold levels and uses warp shuffle for final5stages. Same synchronized output. No scored replay. |
| `MOJOLEARN_PSHAP_DELTA` / `_OFF` | permutation-shap / istella | measured lane/apple-fast-w4-shap@20bbdc37256bd00ef09fc97970f79f6967fa4f2a; default lane/apple-fast-pshap-delta-default | w2-pdelta-quality; w2-pdelta-pshap-istella | 28,222.8 -> 14,788.6 ms (-47.6%); linear/tanh/two-output attribution arrays byte-identical, model rows 2713500 ->2215900 | DEFAULT: M2 default/rollback PASS; M3 pshap-delta-default-r1-quality-20261004 PASS, all13arrays exact. No arithmetic change, no host SHAP math, current-main source compatibility verified; callback remains existing boundary. See ab/pshap-delta-default.md. |

## Catalog shared GEMM screen and actual caller expansion (2026-10-04)

Catalog9ab2d3d3f intake is retained; all10 variants compiled at9f1a1657c.
M3 catalog-matrix-t-v1, tools7dfae82b6: 66 one-call records, six predeclared
n>=2 shapes, all output hashes exactly match incumbent. Unrestricted quality
remains HOLD for the n1 GEMV regression; this scope excludes n1 explicitly.
Cold context + upload + kernel + download + completion + first read total
is about23–30ms; this is not a pure kernel throughput measurement.
There is no universal winner. G1 denseNN ratio0.969, squareNN0.999 and
low-widthNT0.934; G5 tall projection0.932 and Gram0.913; G9 Gram0.885.
No default or caller speed claim follows from this screen. No scored replay.
Report: ~/mojolearn-evidence/apple-fast/sync/catalog-matrix-t-v1-report.json.

The shared G1/G5 dispatcher a9e64922f passed43 route/quality fixtures.
Actual estimator counter/quality harness0505c9427 is source-only: core and
estimators first; zero route reach cannot pass admission. A distinct resident
input contract is being prepared to exclude setup/uploads while retaining
completion and first read; new lifecycle quality must pass before timing.
Decomp/PCA adapters must preserve caller-specific split plans, strides and
atomic behavior. PCA still uses atomic split mode even with one split.

SHAP default merged cf46d6f37 after M2 default/OFF and final M3 quality PASS.
KNN GPU-only lean counts default76567110e M2 default/OFF PASS; final M3
quality remains pending. Callpath r3 failed a context identity guard before
numerical evaluation; r4 14761655f retains stable context ownership and
foreign-context refusal, now compiling on M2. No callpath timing admitted.

Eigh panel DF21eeacf90: M3 eigh-panel-df-q-v1 strict no-regression FAIL against current panel default, absolute gate FAIL; no timings or promotion admitted. K1 diagnostic1a58b974e is reference-only with22fixed evaluations across3explicit model controls; original K1 HOLD remains.

Shared downstream G1 source30e4562c2129569ed03d93d878ec6a903ea51691
starts M2 core A=COUNTERS, B=COUNTERS+G1; G5 source495c30c33a8805a1944b45f9bc911446f7e89ed8
is separate to prevent manifest collisions. Core artifact is _mojolearn.so.
No runtime admission yet. Resident catalog probe fa39073607e3b19c4fbfe063a5545a63318f13c2
starts M2 A/B compilation; strict new lifecycle/matrix quality must pass before
its distinct resident-input call/completion/first-read timing contract.
Callpathr5 3491a4d4cab94ad76a04e779952daa8ba64ef1c8 M2bothPASS; M3qualityowed.
EighDF failure details: board4096 eigenerror +34.3%, board1000 residual+0.65%,
indefinite1024 orthogonality+0.85%; five fallback cases unchanged. HOLD retained.

| `MOJOLEARN_XN_FAST_NAN_FIT_LEAN_GPU` / `_OFF` | knn-imputer / taxi | measured f9c5887a8ff3b5fe8878e0bfe3d5f553a7f741f7; default lane/apple-fast-knn-lean-gpu-default, base1f33e9764 | w2-knn-lean-gpu-taxi-r1; repaired fixtureknn-lean-gpu-v2 | 2.2 ->0.8 ms; quality31 arrays byte-exact | DEFAULT: M2 default/rollback PASS, final M3 knn-lean-gpu-default-quality-20261004 PASS31exact arrays, Python import PASS. GPU count/total reduction and pool reuse, no imported host-lean or unrelated old candidates. See ab/knn-lean-gpu-default.md. |

G1core+estimators30e4562c M2botharmsPASS, staged. Tools-only helperef2ce40dd
fixes KMeans squared-distance oracle and validates compiledancestor/runtime
identity; first unscored cases OLS/PCA/wide-kKNN/KMeans atwidth65 targetcoreNT
route0. Small-kKNNwidth11 is a NO_REACH control. Other shared routes require
separate eligible callers; no claim that all four routes are covered here.
Residentcatalogfa390736 M2botharmsPASS/staged, fresh lifecycle quality queued.
