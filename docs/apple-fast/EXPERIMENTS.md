# Apple FAST experiments: every A/B, kept and dropped

The record of every Apple FAST (M3 Ultra) experiment from Oct 2 to Oct 3, 2026: the winners that became defaults, the losers, and the A/Bs still owed.
Sources: `~/mojolearn-evidence/apple-fast/LEDGER.md` (every KEEP / DROP / MERGED / CANDIDATE line) and the queue lines (`docs/apple-fast/ab/*.txt`) and notes (`docs/apple-fast/notes/*.md`) on each lane branch.

## How to use it

- **Before you write a new experiment, search this file for the define** (and for the algorithm). If it was dropped, read the reason first. Do not re-run a dropped idea unless the code it touched has changed since.
- **To get dropped code back**, use the branch and sha in its row: `git show <sha>:<file>`, or `git diff origin/main...<sha> -- <dir>` for the whole change. The lane branches stay on origin.
- **Before/after** is milliseconds on the M3 Ultra, FAST mode. Arm A is FAST main code; arm B adds the define. n=1 or n=2 per arm, as the ledger notes. Board numbers are in `BOARD_M3_FAST.md`.
- **Lesson (Oct 3):** an A/B on a branch whose base is older than main can show a gain main already has. Judge such a gain against main's board, or re-run the A/B with main merged in.

## Policy

- **Winners** become FAST + Apple defaults. Each one gets a `<NAME>_OFF` define that restores the old path, and a code comment citing its A/B (tag and numbers). IDENTICAL mode never changes.
- **Losers are not kept in main's code.** They stay on their branch at the recorded sha. A dropped define that is still on main is a dead toggle; a cleanup lane removes those.
- Quality comes before speed: a faster arm with worse quality is DROPPED-quality, and a slower arm with better quality can be KEPT (see `X_PREP_CLASS_COV_GRID`).

## Verdicts

| verdict | meaning | rows |
|---|---|---|
| KEPT `<main sha>` | FAST + Apple default since that main commit | 85 |
| DROPPED-slower | B slower than A | 19 |
| DROPPED-noise | the difference is inside run-to-run spread (arms overlap, or under 5% at n=1), or the signs are mixed across datasets | 47 |
| DROPPED-quality | faster, but the quality metric got worse | 2 |
| DROPPED-semantics | no longer matches main's code path: stale base, duplicate of a main change, a no-op, or a host step in the GPU path | 4 |
| OPEN | not measured yet, measured but not judged, or the run did not finish | 270 |

Each row is one define, or one combination of defines, on one branch. Combination rows (`A + B`) are B arms that turn on several defines together. `(_OFF)` in a define name means the switch on main is the opt-out. Shas are `git rev-parse --short origin/<branch>` as of this record; the ledger's measured head is given where it differs.

## Trees (101)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
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
| `ET_DEVICE_BATCH_65536` | et / taxi | lane/apple-fast @ 269ffa57a | aft-ab-etb64, aft-ab-etb64b | on PART_ROWS: taxi 3,041 -> 3,080 | DROPPED-slower | +1.3% |
| `ET_TPB_256` | et / taxi, istella | lane/apple-fast @ 269ffa57a | aft-ab-ettpb, aft-ab-ettpb2 | on PART_ROWS: taxi 3,030 -> 3,072; istella 4,133 -> 4,379 | DROPPED-slower | +1.4% / +6% (alone it was mixed) |
| `GBDT_CTR_FAST_FREQ + GBDT_CTR_FAST_SCAN` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-both | categorical 32,739 -> 27,662 | DROPPED-noise | same as FREQ alone; SCAN adds nothing |
| `GBDT_CTR_FAST_SCAN` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-scan | categorical 32,892 -> 33,044 | DROPPED-noise | +0.5%, overlap |
| `GBDT_CTR_PERM_PTRS` | categorical | lane/apple-fast-trees-depthwise @ f743edd60 | tdw-cat-permptrs | categorical taxicat 37,098 -> 36,618 | DROPPED-noise | -1.3%, B runs straddle A |
| `GBDT_DW2_COPY_ZERO` | depthwise / istella, taxi | lane/apple-fast-dwgap2 @ 23eca012b | dw2-copy-zero-taxi, dw2-copy-zero-istella | depthwise taxi 13,051 -> 13,441 | DROPPED-slower | +3.0%, arms overlap |
| `GBDT_DW_FAST_DEV_SCALE` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-devscale-dwtaxi | 10,812 -> 10,780 | DROPPED-noise | -0.3%, overlap |
| `GBDT_DW_FAST_SKIP_FINAL_STATS` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-skipfs-dwtaxi | 11,108 -> 11,048 | DROPPED-noise | -0.5% |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC` | depthwise / taxi | lane/apple-fast-depthwise @ 4547e0d14 | dw-tree-taxi | depthwise taxi 14,453 -> 14,386 | DROPPED-noise | -0.5%, B runs straddle A |
| `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC + GBDT_DW_TREE_SYNC_CHECK` | depthwise / taxi | lane/apple-fast-depthwise @ 4547e0d14 | dw-tree-check-taxi | depthwise taxi 14,485 -> 14,581 | DROPPED-noise | +0.7% |
| `GBDT_LG_EXACT_BATCH16` | lossguide / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-lgw16 | taxi 16,776 -> 18,093 | DROPPED-slower | +8% |
| `GBDT_QH_FAST_FUSED_Q` | depthwise / taxi | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-fusedq-dwtaxi | 10,673 -> 10,783 | DROPPED-noise | +1%, overlap |
| `GBDT_SEG_SUMS_BLOCK` | depthwise / taxi | lane/apple-fast-rfet-scan @ 500168cfe | aft-ab-gbseg | 10,325 -> 10,293 | DROPPED-noise | -0.3%, overlap; stays opt-in |
| `GBDT_SM_X4` | depthwise / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-smx4 | taxi 14,881 -> 14,772 | DROPPED-noise | arms overlap |
| `GBDT_SM_X8` | lossguide / taxi | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-smx8 | taxi 16,483 -> 19,577 | DROPPED-slower | +19% |
| `IF_QUERY_RAW` | iforest | lane/apple-fast-trees-io @ df6a77c21 | trees-io-ifq-build | iforest taxi 328 -> 324; score istella 2,290 -> 2,289 | DROPPED-noise | A runs straddle; refusals ok |
| `IF_SAMPLED_UPLOAD` | iforest / istella | lane/apple-fast-trees-io @ df6a77c21 | trees-io-if-istella | iforest istella 384 -> 385 | DROPPED-noise | same hash |
| `ORDERED_FOLD_DERIVS` | ordered / taxi | lane/apple-fast-ordered @ 5b8722353 | ord-fd-taxi | ordered taxi 282,245 -> 287,217 | DROPPED-slower | +1.8% |
| `REORDER_FLAGS_SCAN_BLOCK + SEG_SCAN_BLOCK` | depthwise / rf / taxi | lane/apple-fast-trees-scan @ 43430ca0f | trees-scan-dw-taxi | depthwise taxi 12,631 -> 12,853 | DROPPED-noise | +1.8%, overlap |
| `RF_NODESPLIT_ZERO_AFTER_READ + RF_FAST_BATCH16K` | rf / taxi, istella | lane/apple-fast-trees2 @ bfd1d7cc6 | aft-ab-rf1 | taxi 11,502 -> 11,485; istella 14,324 -> 14,312 | DROPPED-noise | 0.1% |
| `RF_SMALL_NODE_1024` | rf / taxi, istella | lane/apple-fast-trees2 @ 50dfdcca0 | aft-ab-rfsn | taxi 11,491 -> 11,482; istella 14,320 -> 14,329 | DROPPED-noise |  |
| `SEG_SCAN_BLOCK` | depthwise / rf / istella | lane/apple-fast-trees-scan @ 43430ca0f | trees-scan-rf-istella | rf istella 13,704 -> 10,146 | DROPPED-semantics | stale base: main already had the gain via SEG_SUMS_BLOCK_SCAN (69f7a41fd); duplicate |
| `SYM_DEVICE_LEAVES` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-leaves-istella | symmetric istella 14,671 -> 14,643 | DROPPED-noise | -0.2%, overlap |
| `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-all-1000-istella | symmetric-1000 istella 32,932 -> 33,146 | DROPPED-noise | +0.6%; no switch in this lane wins |
| `SYM_DEVICE_LEVEL` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-level-istella | symmetric istella 16,926 -> 16,852 | DROPPED-noise | -0.4%; auc equal |
| `SYM_DEVICE_PARTITION` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-part-istella | symmetric istella 16,869 -> 17,081 | DROPPED-slower | +1.3% |
| `SYM_HIST_FAST` | yetirank / symmetric / istella, taxi | lane/apple-fast-trees-yeti @ 65f551e39 | yeti-symhist, yeti-symhist-sym-taxi, yeti-symhist-sym-istella, yeti-symhist-ordered-taxi | symmetric istella 14,710 -> 14,685; ordered taxi 66,705 -> 67,150 | DROPPED-noise | -0.2% / +0.7% |
| `SYM_HIST_FAST + YETI_SEARCH_TASK16K` | yetirank / symmetric | lane/apple-fast-trees-yeti @ 65f551e39 | yeti-both | - | DROPPED-noise | SYM_HIST_FAST part dropped (see above) |
| `SYM_HIST_FAST + YETI_SYM_HIST_UNROLL8` | yetirank | lane/apple-fast-yetirank @ c7b35fd7c | yeti-h8unroll | yetirank 5,315 -> 5,373 | DROPPED-slower | +1.1% |
| `SYM_NO_TAIL_DRAIN` | symmetric / istella | lane/apple-fast-trees-symmetric @ ce517b4b3 | tsym-notail-istella | symmetric istella 16,998 -> 17,106 | DROPPED-noise | +0.6% |
| `YETI_TREE_SEARCH_SCORE_GRID` | yetirank | lane/apple-fast-yetirank @ c7b35fd7c | yeti-scoregrid | yetirank 5,304 -> 5,288 | DROPPED-noise | -0.3%, overlap |
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
| `ORD_ALL` | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-all-taxi, sym-ordered-all-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ORD_FOLD_BINS_ONE` | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-fbo-taxi, sym-ordered-fbo-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ORD_FOLD_INDEX` | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-fidx-taxi, sym-ordered-fidx-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ORD_STD_PARALLEL` | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-std-taxi, sym-ordered-std-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `ORD_TREE_LEAN` | ordered / istella, taxi | lane/apple-fast-sym-ordered @ 27b912397 | sym-ordered-lean-taxi, sym-ordered-lean-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
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
| `KERNEL_FAST_BAYES_JACOBI` | bayesian-ridge / istella | lane/apple-fast-kernel @ 9e851777c | kernel-bayes-jacobi-ist | bayesian-ridge istella 2,741 -> 2,726 | DROPPED-noise | -0.6% |
| `KERNEL_FAST_BAYES_STATS` | bayesian-ridge / istella | lane/apple-fast-kernel @ 9e851777c | kernel-bayes-stats-ist | bayesian-ridge istella 2,748 -> 3,143 | DROPPED-slower | +14.4% |
| `OLS_FAST_DEVICE_CENTER` | ols / taxi | lane/apple-fast-core @ 9a31ebb4c | core-ols-dcenter-taxi | ols taxi 335.6 -> 157.6 | DROPPED-semantics | stale base: main OLS route changed (TSQR, then normal eq) |
| `QN_FAST_COALESCED_OFF` | logreg / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-logreg-nocoal-istella | logreg istella 3,889 -> 4,996 | DROPPED-slower | +28.5% (turning coalescing off) |
| `QN_FAST_GRID_SUMS` | linearsvc / istella; linearsvr / istella; logreg / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-logreg-gs-istella, linear-svc-gs-istella, linear-svr-gs-istella | logreg 3,890 -> 4,235; svc 775 -> 802; svr 914 -> 213 | DROPPED-noise | mixed signs |
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
| `LSVR_ALL` | linearsvr / istella; linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-all-taxi, linsvr-all-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_DEVICE_CONVERGE` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dconv-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dconv-vs-batch-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_DUAL_CD` | linearsvr / istella; linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-dualcd-taxi, linsvr-dualcd-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_EVAL_SLIM` | linearsvr / istella; linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-slim-taxi, linsvr-slim-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_FASTPATH_FIX` | linearsvr / istella; linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-fix-taxi, linsvr-fix-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_FUSED_GRAD` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-fused-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `LSVR_LINESEARCH_BATCH` | linearsvr / taxi | lane/apple-fast-linsvr @ c649076a4 | linsvr-lsbatch-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `NB_CAT_ATOMIC` | categorical-nb / taxi | lane/apple-fast-nb @ be2ea3a05 | nb-cat-atomic-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `NB_TEXT_CSR` | complement-nb / text; multinomial-nb / text | lane/apple-fast-nb @ be2ea3a05 | nb-mnb-csr-text, nb-cnb-csr-text | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `RIDGE_FAST_CLS1_CODES` | ridge-clf / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-rccodes-taxi | ridge-clf taxi 120 -> 19.0 | OPEN | judged KEEP (-84%); merge pending |

## Neighbors (39)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `KDE_FAST_SLICES` | kde / istella | lane/apple-fast-core @ 9a31ebb4c | core-kde-slices-istella | kde istella 1,519 -> 141 | KEPT 857fd5804 | -91%; log-likelihood same |
| `KNN_FAST_MMA_BIGD` | knn-clf, knn / istella | lane/apple-fast @ 269ffa57a | afc-knn2 | knn-clf 864 -> 290; kneighbors 1,727 -> 1,788 | KEPT 269ffa57a | knn-clf 3.0x; kneighbors noise at n=1 |
| `KNN_FAST_MMA_K64` | knn / istella | lane/apple-fast-core @ 9a31ebb4c | core-knn-k64-istella | knn istella 1,581 -> 352 | KEPT 857fd5804 | -78%; recall same |
| `LP_FAST_RESIDENT` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-lp-res-taxi, n2-ls-res-taxi | label-propagation taxi 6,793 -> 2,086; label-spreading 2,095 -> 2,004 | KEPT 1e288d323 | -69% (comptime default) |
| `KNN_FAST_CLS1_PRESEED` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnseed-istella | knn istella 357 -> 343 | DROPPED-noise | -3.9%, n=1, marginal |
| `KNN_FAST_CLS1_PRESEED + KNN_FAST_CLS1_SLICES2` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnall-istella | knn istella 357 -> 385 | DROPPED-slower | slower |
| `KNN_FAST_CLS1_SLICES2` | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-knnsl2-istella | knn istella 357 -> 421 | DROPPED-slower | slower |
| `LLE_FAST_KNN` | - | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-lle-knn-taxi | lle taxi 4,266 -> 4,289 | DROPPED-noise | +0.5% |
| `NC_FAST_CLS1_LABELS + NC_FAST_CLS1_PREDICT` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ncall-taxi | nearest-centroid taxi 105 -> 27 | DROPPED-noise | worse than LABELS alone |
| `NC_FAST_CLS1_PREDICT` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-ncpred-taxi | nearest-centroid taxi -4% | DROPPED-noise | <5% |
| `RADIUS_FAST_REUSE_COUNT` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-radius-reuse-taxi | radius-neighbors taxi 0.1 -> 0.1 | DROPPED-noise | ms-scale |
| (baseline, no switch) | knn / istella | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-k64chk-istella | knn istella (main) 357 | OPEN | baseline re-time only (K64 default); row flipped faster than sklearn 566 |
| `ANN_FAST_KNN_BIGD` | cagra / istella | lane/apple-fast-ann @ 70833546a | ann-cagra-knnbigd-istella | - | OPEN | A/B queued, no judged result yet |
| `CAGRA_FAST_TEAM` | cagra / istella | lane/apple-fast-ann @ 70833546a | ann-cagra-team-istella | - | OPEN | A/B queued, no judged result yet |
| `ISOTONIC_FAST_PAIRMERGE + ISOTONIC_FAST_PAR` | isotonic / istella | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-iso-pair-istella | - | OPEN | A/B queued, no judged result yet |
| `ISOTONIC_FAST_PAR` | isotonic / istella | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-iso-par-istella | - | OPEN | A/B queued, no judged result yet |
| `IVFPQ_FAST_DEVICE_CODEBOOKS` | ivf-filter / istella; ivf-pq / istella; ivf-refine / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-devcb-istella, ann-ivfrefine-devcb-istella, ann-ivffilter-devcb-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_COARSE_RANDOM_INIT` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-rinit-istella, vsearch-ivf-rinit-istella, vsearch-ivf-rinit-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_DEVICE_VALIDATE` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-dval-istella, vsearch-ivf-dval-istella, vsearch-ivf-dval-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_FAST_DEVICE_CSR` | ivf / istella; ivf-pq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-csr-istella, ann-ivf-csr-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_FAST_DEVICE_TRAINSET` | ivf / istella; ivf-pq / istella; ivf-sq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-trainset-istella, ann-ivfsq-trainset-istella, ann-ivf-trainset-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_FAST_SCAN_SELECT` | ivf-pq / istella; ivf-rabitq / istella; ivf-sq / istella | lane/apple-fast-ann @ 70833546a | ann-ivfpq-select-istella, ann-ivfsq-select-istella, ann-ivfrq-select-istella | - | OPEN | A/B queued, no judged result yet |
| `IVF_KMEANS_LAZY_SHIFT` | ivf / istella; ivf / taxi; ivf-pq / istella | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-lazy-istella, vsearch-ivf-lazy-istella, vsearch-ivf-lazy-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `IVF_REFINE_TEAM` | ivf-refine / istella; ivf-refine / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-refine-team-istella, vsearch-refine-team-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE2_ALL` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-all-istella, kde2-all-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE2_ALL + KDE_DIMTILE` | kde / istella | lane/apple-fast-kde2 @ 659400b94 | kde2-all-vs-dimtile-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_DIMTILE` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-dimtile-istella, kde2-dimtile-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_DIMTILE + KDE_KERNEL_VARIANTS` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-variants-istella, kde2-variants-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_DIMTILE + KDE_LSE_FUSED` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-lse-istella, kde2-lse-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_DIMTILE + KDE_NORM_FUSED` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-norm-istella, kde2-norm-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_DIMTILE + KDE_SAMPLE_FUSED` | kde / istella | lane/apple-fast-kde2 @ 659400b94 | kde2-sample-ontile-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `KDE_SAMPLE_FUSED` | kde / istella; kde / taxi | lane/apple-fast-kde2 @ 659400b94 | kde2-sample-istella, kde2-sample-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `NC_FAST_CLS1_LABELS` | nearest-centroid / taxi | lane/apple-fast-gap-cls1 @ 4e341dc41 | gapcls1-nclabels-taxi | nearest-centroid taxi 105 -> 21 | OPEN | judged KEEP (-80%, acc same); cls1 merge pending |
| `PQ_LUT_TILED` | ivf-pq / istella; ivf-pq / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-lut-istella, vsearch-pq-lut-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PQ_SCAN_FUSED` | ivf-filter / istella; ivf-pq / istella; ivf-pq / taxi | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-fused-istella, vsearch-filter-fused-istella, vsearch-pq-fused-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `TSNE_FAST_SPLIT` | tsne / istella | lane/apple-fast-ann @ 70833546a | ann-tsne-split-istella | - | OPEN | A/B queued, no judged result yet |
| `VSEARCH_ALL` | ivf / istella; ivf / taxi; ivf-filter / istella; ivf-filter / taxi; ivf-pq / istella; i... | lane/apple-fast-vsearch @ 86925aef9 | vsearch-pq-all-istella, vsearch-ivf-all-istella, vsearch-ivf-all-taxi (+8) | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `XN_FAST_IMPUTE_TILED2` | knn-imputer / taxi | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-imp-t2-taxi | - | OPEN | A/B queued, no judged result yet |
| `XN_FAST_MMA_ROUTE` | lle / taxi; lof / taxi | lane/apple-fast-isotonic-knn @ 7385fcfdd | ik-lle-mma-taxi, ik-lof-mma-taxi | - | OPEN | A/B queued, no judged result yet |

## Prep (38)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `PREP_FAST_MINMAX` | minmax-scaler / istella | lane/apple-fast-prep @ a11e43a5e | prep-minmax-istella | minmax-scaler istella 152 -> 106 | KEPT f419ea9f1 | -30%; output digest identical |
| `SELECT_D` | select-d / taxi-hourly | lane/apple-fast-select @ 99fad7a5d | sel-d-hourly | select-d taxi-hourly 7.8 -> 3.9 | KEPT e2bfb8422 | -50%; digest identical |
| `SELECT_FCLS` | select-f-classif / taxi | lane/apple-fast-select @ 99fad7a5d | sel-fcls-taxi | select-f-classif taxi 131.4 -> 17.5 | KEPT e2bfb8422 | -87% |
| `SELECT_FREG` | select-f-regression / taxi; select-r-regression / taxi | lane/apple-fast-select @ 99fad7a5d | sel-freg-taxi, sel-rreg-taxi | select-r-regression taxi 100.5 -> 10.1; select-f-regression 102.6 -> 15.0 | KEPT 4198d5a9c | -90% / -85% |
| `X_PREP_FAST_UNIQUE` | - | lane/apple-fast-prep @ a11e43a5e | prep-onehot-uniq-taxi, prep-ordinal-uniq-taxi | onehot taxi 69.7 -> 30.4; ordinal taxi 67.2 -> 25.3 | KEPT f419ea9f1 | -56% / -62% |
| `X_PREP_FAST_NONEG` | - | lane/apple-fast-prep @ a11e43a5e | prep-onehot-noneg-taxi, prep-ordinal-noneg-taxi | onehot -2.1%; ordinal -0.1% | DROPPED-noise | <5% |
| `CV_FAST_SLICE` | cross-val-score / taxi | lane/apple-fast-resample @ 50b96e795 | resample-cv-slice-taxi | - | OPEN | A/B queued, no judged result yet |
| `CV_FAST_TRUST_FOLDS` | cross-val-score / taxi | lane/apple-fast-resample @ 50b96e795 | resample-cv-trust-taxi | - | OPEN | A/B queued, no judged result yet |
| `MI_ALL` | select-mutual-info / istella; select-mutual-info / taxi; select-mutual-info-reg / istel... | lane/apple-fast-mi @ 6944ebb57 | mi-reg-all-istella, mi-reg-all-taxi, mi-clf-all-istella, mi-clf-all-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_CLF_RANKMAJOR` | select-mutual-info / istella; select-mutual-info / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-clf-rankmajor-istella, mi-clf-rankmajor-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_FAST_FOLDS` | select-mutual-info / istella; select-mutual-info / taxi; select-mutual-info-reg / istel... | lane/apple-fast-mi @ 6944ebb57 | mi-reg-folds-istella, mi-reg-folds-taxi, mi-clf-folds-istella, mi-clf-folds-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_REG_RANKMAJOR + MI_REG_SORTCOUNT` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-rankmajor-istella, mi-reg-rankmajor-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_REG_SORTCOUNT` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-sortcount-istella, mi-reg-sortcount-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_REG_SORTCOUNT + MI_REG_TIES` | select-mutual-info-reg / istella; select-mutual-info-reg / taxi | lane/apple-fast-mi @ 6944ebb57 | mi-reg-ties-istella, mi-reg-ties-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MI_WORK` | - | lane/apple-fast-mi @ 6944ebb57 | mi-reg-work-istella, mi-reg-work-taxi, mi-clf-work-istella, mi-clf-work-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP2_FAST_EIGH_BLOCK` | iterative-imputer / taxi | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-eigh-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP3_LABELS` | label-binarizer / taxi; label-encoder / istella; multilabel-binarizer / taxi | lane/apple-fast-prep3 @ ec65873e3 | prep3-le-istella, prep3-lb-taxi, prep3-mlb-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP3_MAXABS` | maxabs-scaler / istella | lane/apple-fast-prep3 @ ec65873e3 | prep3-maxabs-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP3_SPLINE` | spline / istella | lane/apple-fast-prep3 @ ec65873e3 | prep3-spline-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PREP_FAST_CLS2_MINMAX_FUSED / _MINMAX_POOL` | minmax-scaler | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |
| `PTIMPUTE_ALL` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-all-istella, ptimpute-pt-all-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PT_COLBATCH` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-colbatch-istella, ptimpute-pt-colbatch-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PT_COLBATCH + PT_SPEC` | power-transformer / istella | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-spec-vs-colbatch-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PT_FOLD_NOX` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-nox-istella, ptimpute-pt-nox-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PT_FUSED_TRANSFORM` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-fusedtx-istella, ptimpute-pt-fusedtx-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `PT_SPEC` | power-transformer / istella; power-transformer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-spec-istella, ptimpute-pt-spec-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `RESAMPLE_FAST_GATHER` | resample / taxi | lane/apple-fast-resample @ 50b96e795 | resample-rs-gather-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_IDX_BULK` | resample / taxi | lane/apple-fast-resample @ 50b96e795 | resample-rs-idxbulk-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_ONE_FOLD` | bootstrap / taxi | lane/apple-fast-resample @ 50b96e795 | resample-boot-onefold-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_PERM_SELECT` | permutation-test / taxi | lane/apple-fast-resample @ 50b96e795 | resample-perm-select-taxi | - | OPEN | A/B queued, no judged result yet |
| `RESAMPLE_FAST_RANK_SORT` | bootstrap / taxi | lane/apple-fast-resample @ 50b96e795 | resample-boot-rank-taxi | - | OPEN | A/B queued, no judged result yet |
| `SI_ONEPASS` | power-transformer / istella; simple-imputer / istella; simple-imputer / taxi | lane/apple-fast-ptimpute @ 9623cd7dc | ptimpute-pt-onepass-istella, ptimpute-si-onepass-istella, ptimpute-si-onepass-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `X_PREP_FAST_CLS2_PACK / _PRESENT` | onehot, ordinal | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |
| `X_PREP_FAST_II_CONV` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-conv-taxi | both arms status=error | OPEN | env-form line; define-form -b relaunch pending |
| `X_PREP_FAST_II_GRAM_TILE` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-ii-gram-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_QSELECT` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-si-qsel-istella, prep2-rs-qsel-istella | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_TE_ENC` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-te-enc-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |
| `X_PREP_FAST_TE_GLOBAL` | - | lane/apple-fast-prep2 @ 8762eb33f | prep2-te-global-taxi | both arms status=error | OPEN | env-form line; -b relaunch pending |

## Decomp (31)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `APPLE_FAST_GEMM_TN_V1` | ols | lane/apple-fast-tier @ 95a09d1fd | tier-pca-tnv1, tier-ols-tnv1 | ols istella 2,219 -> 935 | KEPT 7484503b6 | -58%; r2 .3211 -> .3319 (re-measured as tier-ols-tnv1b, define form) |
| `LDA_FUSED_SS` | lda / taxi-zones | lane/apple-fast-nb @ be2ea3a05 | nb-lda-fused-zones | lda taxi-zones 2,721 -> 1,509 | KEPT a7b8b9513 | -44.5%; perplexity same |
| `APPLE_FAST_GEMM_NT_TILED` | kmeans | lane/apple-fast-tier @ 95a09d1fd | tier-pca-nttiled, tier-ols-nttiled, tier-kmeans-nttiled | ols istella 2,179 -> 2,666; pca istella 599.6 -> 601.1; kmeans istella 1,474 -> 1,523 | DROPPED-slower | never wins |
| `DECOMP_FAST_SMALL_EIGH_J2` | fastica / istella | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-ica-istella | - | DROPPED-semantics | no-op after main removed _eigh2 |
| `LLE_SPARSE_EIG` | lle / taxi | lane/apple-fast-lle @ f2ea1ecb5 | lle-sparse-taxi | lle taxi 4,274 -> 9,174 | DROPPED-slower | +115% |
| `PCA_FAST_EIG` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-eig-istella, pca-eig-eig-taxi, pca-eig-tsvd-eig-istella (+2) | pca istella -2.2% / taxi +17%; tsvd 0%; ipca +0.4% / -1.2% | DROPPED-noise | mixed signs across datasets |
| `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` | pca / tsvd / ipca / istella | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-all-istella | pca istella -0.7% | DROPPED-noise | noise |
| `PCA_FAST_NO_ALIAS` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-noalias-istella, pca-eig-noalias-taxi | pca istella +2.1% / taxi -8.5% | DROPPED-noise | mixed signs |
| `PCA_FAST_TOPK` | pca / tsvd / ipca / istella, taxi | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-topk-istella, pca-eig-topk-taxi, pca-eig-tsvd-topk-istella | pca istella +1.5% / taxi -5.9%; tsvd +0.5% | DROPPED-noise | mixed signs |
| (baseline, no switch) | pca / tsvd / ipca | lane/apple-fast-pca-eig @ 9819970b1 | pca-eig-ident | - | OPEN | IDENTICAL baseline line |
| (baseline, no switch) | pca | lane/apple-fast-tier @ 95a09d1fd | tier-pca-ident, tier-rbf-ident, tier-theta-ident, tier-theta-fast | - | OPEN | IDENTICAL / FAST baseline lines |
| `CHOL_FAST_BLOCKED` | cholesky / synthetic | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-chol-blocked-synthetic | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `CHOL_FAST_BLOCKED + SVD_FAST_CHOLQR` | svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-svd-cholqr-chol-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_DICT_UPDATE` | dict-learning / istella; mb-dict-learning / istella; mb-sparse-pca / istella; sparse-pc... | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-mbdl-upd-istella, dsp-dl-upd-istella, dsp-mbspca-upd-istella, dsp-spca-upd-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_GEMM_TILED` | als / taxi-zones; lstsq / istella; nmf / istella; randomized-svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-lstsq-tiled-istella, dlin-rsvd-tiled-istella, dlin-nmf-tiled-istella, dlin-als-tiled-taxizones | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `DECOMP_FAST_LASSO_BLOCK` | dict-learning / istella; mb-dict-learning / istella; sparse-pca / taxi | lane/apple-fast-decomp-sparse @ 5fb1740cd | dsp-mbdl-lasso-istella, dsp-dl-lasso-istella, dsp-spca-lasso-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
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
| `MCD_DEVICE_CSTEPS` | elliptic-envelope / taxi; min-cov-det / taxi | lane/apple-fast-robust @ cfdb95e48 | robust-mcd-taxi, robust-ee-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `QR_FAST_DEV` | qr / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-qr-dev-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `SVD_FAST_CHOLQR` | svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b | dlin-svd-cholqr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `XD_FAST_CLS2_GRP_DEVSCAN` | gaussian-rp / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | 54 -> 15.9 | OPEN | judged KEEP (-71%, keeps fit-time NaN refusal; beats sklearn 24.3); cls2 merge pending |
| `XD_FAST_CLS2_GRP_NOSCAN (+ _GRP_LAZY)` | gaussian-rp / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | 57 -> 1.1; with LAZY 54 -> 0.5 | OPEN | semantics: moves the NaN/inf error from fit to transform; Andrew asked, not default |

## Cluster (35)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `GMM_FAST_BIG_CHOL` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-bc-istella | gmm istella 6,560 -> 5,894 | KEPT 0ae9c28cd | -10.2%; bic same |
| `MBK_ZEROCOPY` | minibatch-kmeans / istella, taxi | lane/apple-fast-mbkspeed @ 44e6d93e6 | mbkzc-istella, mbkzc-taxi | istella 351 -> 256; taxi 51.1 -> 48.5 | KEPT 51493c7a8 | -27% / -5% |
| `X_CLUSTER_FAST_MEANSHIFT` | meanshift / istella | lane/apple-fast-cluster @ f707846c8 | cluster-ms-istella | meanshift istella 52.1 -> 34.9 | KEPT 4b1311c12 | -33%; same clusters |
| `X_CLUSTER_FAST_MINIBATCH` | minibatch-kmeans / istella | lane/apple-fast-cluster @ f707846c8 | cluster-mb-istella | minibatch-kmeans istella 388 -> 352 | KEPT 4b1311c12 | -9.5%; silhouette .1167 -> .1182 |
| `GMM_FAST_BIG_CHOL + GMM_FAST_ESTEP_STACK + GMM_FAST_GRID_COV` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-all-istella | gmm istella 6,587 -> 9,046 | DROPPED-slower | +37.3% |
| `GMM_FAST_ESTEP_STACK` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-es-istella | gmm istella 6,566 -> 6,886 | DROPPED-slower | +4.9% |
| `GMM_FAST_GRID_COV` | gmm / istella | lane/apple-fast-linear @ 1c7c213f8 | linear-gmm-gc-istella | gmm istella 6,584 -> 9,858 | DROPPED-slower | +49.7%; n_iter 24 -> 42 |
| `KMEANS_FAST_ROWNORM` | kmeans / taxi | lane/apple-fast-core @ 9a31ebb4c | core-kmeans-rownorm-taxi | kmeans taxi -3.1% | DROPPED-noise | <5% at n=1; inertia same |
| `KMEANS_FAST_SKIP_PREDICT` | kmeans / taxi | lane/apple-fast-core @ 9a31ebb4c | core-kmeans-skippred-taxi | kmeans taxi -2.2% | DROPPED-noise | <5% at n=1 |
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
| `HDBSCAN2_ALL` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-all-taxi, hdbscan2-all6-taxi, hdbscan2-all-istella, hdbscan2-all6-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_CORE_TILE` | hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-core-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_DEV_BORUVKA` | hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-boruvka-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_LINKAGE_DEVICE` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-linkage-taxi, hdbscan2-linkage-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_ONE_SYNC` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-onesync-taxi, hdbscan2-onesync-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_SELECT_DEVICE` | hdbscan / istella; hdbscan / taxi | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-select-taxi, hdbscan2-select-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `HDB_SMR_TILED` | hdbscan / istella | lane/apple-fast-hdbscan2 @ 2fdb9114f | hdbscan2-smr-istella | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `OPTICS2_ALL` | optics / istella; optics / taxi | lane/apple-fast-optics2 @ 3e192fdb7 | optics2-all-istella, optics2-all-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `OPTICS_CORE_SQ` | optics / istella; optics / taxi | lane/apple-fast-optics2 @ 3e192fdb7 | optics2-sq-istella, optics2-sq-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `OPTICS_FAST_DEVICE_ORDER` | optics / istella | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-optics-devorder-istella | - | OPEN | A/B queued, no judged result yet |
| `OPTICS_FRONTIER_DEVICE` | optics / istella; optics / taxi | lane/apple-fast-optics2 @ 3e192fdb7 | optics2-fd-istella, optics2-fd-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `OPTICS_LIVEBUF` | optics / istella; optics / taxi | lane/apple-fast-optics2 @ 3e192fdb7 | optics2-lb-istella, optics2-lb-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `OPTICS_STEP_BATCH` | optics / istella; optics / taxi | lane/apple-fast-optics2 @ 3e192fdb7 | optics2-sb-istella, optics2-sb-taxi | - | OPEN | A/B queued (lane/apple-fast-batch prebuilt arms) |
| `X_CLUSTER_FAST_CLS2_MBK_G128 / _MBK_FIN / _MBK_POOL` | minibatch-kmeans / istella | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN | merge guard: never default MBK_FIN (one-block kernel) |

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
| `ARIMA_FAST_LS_NOREAD` | autoarima / synthetic | lane/apple-fast-gap-arima @ d967c0121 | gaparima-noread-synthetic | 28,578 -> 28,587 | DROPPED-noise | 0% |
| `ARIMA_FAST_P_FIX` | autoarima / taxi-hourly | lane/apple-fast-gap-arima @ d967c0121 | gaparima-pfix-taxi-b, gaparima-pfixasync | 39,550 -> 104,849; with ASYNC 39,570 -> 64,830 | DROPPED-slower | +165% |
| `SEQ_FAST_VAR_ONECOPY + SEQ_FAST_VAR_SPEC` | var / synthetic; var / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-varboth-var-taxi-hourly, gaptsa-varboth-var-synthetic | = VAR_ONECOPY alone | DROPPED-noise | SPEC adds nothing |
| `SEQ_FAST_VAR_SPEC` | var / synthetic; var / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-varspec-var-taxi-hourly, gaptsa-varspec-var-synthetic | var +14% / -2% | DROPPED-slower | deleted from main at merge; recoverable at 5057bee75 |
| `TSA2_KPSS` | kpss / synthetic; kpss / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-tsa2kpss-kpss-taxi-hourly, gaptsa-tsa2kpss-kpss-synthetic | kpss taxi-hourly -73%, synthetic -62% | DROPPED-noise | same gain as TSA_FAST_KPSS_PACK (kept); stays opt-in |
| `TSA2_KPSS` | kpss / taxi-hourly | lane/apple-fast-tsa2 @ b27c8169b | tsa2-kpss-taxi-hourly | compile fail on this branch; on gap-tsa: kpss taxi-hourly -73%, synthetic -62% | DROPPED-noise | same gain as TSA_FAST_KPSS_PACK, which was kept instead; stays opt-in |
| (baseline, no switch) | - | lane/apple-fast-seq @ b185f069f | seq-croston-ident, seq-croston-fast | - | OPEN | IDENTICAL / FAST baseline lines |
| `SEQ_FAST_THETA_HOIST` | theta / taxi-hourly | lane/apple-fast-gap-tsa @ e9da47064 | gaptsa-thetahoist-theta-taxi-hourly | theta taxi-hourly 218 -> 58 | OPEN | -73% alone (at 976a585a0); combo gaptsa-spechoist-theta-taxi-hourly queued; on main, default off |
| `SEQ_GARCH_HOST_MAX` | - | lane/apple-fast-seq @ b185f069f | seq-garch-ident-dev, seq-garch-dev | - | OPEN | env-form line (no-op after define switch) |

## Kernel / GP (11)

| define | algorithm / dataset | branch @ sha | A/B tag | before -> after ms | verdict | reason / note |
|---|---|---|---|---|---|---|
| `APPLE_FAST_GEMM_PINNED` | nys / taxi | lane/apple-fast-tier @ 95a09d1fd | tier-rbf-pinned, tier-rbf-pinned-taxi, tier-nys-pinned | rbf taxi 65.0 -> 63.9; nystroem istella 525.6 -> 522.7 | DROPPED-noise | -1.6% / -0.6% |
| `KAPPROX_DEVICE` | additive-chi2 / istella; skewed-chi2 / taxi | lane/apple-fast-kapprox @ 10d5a7970 | kap-schi2-taxi, kap-achi2-istella | skewed-chi2 taxi 2.8 -> 1.3; additive-chi2 istella 11.0 -> 12.5 | DROPPED-quality | kernel_rel_error .0378 -> .0480 (worse) / slower |
| `KERNEL_FAST_GPR_RESIDENT` | gpr / istella | lane/apple-fast-kernel @ 9e851777c | kernel-gpr-resident-ist | gpr istella 168 -> 133 | DROPPED-semantics | not made default: main resident GPR chain already covers it; lane arm folded on host |
| `SPARSE_RP_DEVICE` | sparse-rp / taxi | lane/apple-fast-kapprox @ 10d5a7970 | kap-srp-taxi | sparse-rp taxi 4.9 -> 1.2 | DROPPED-quality | mean_abs_distortion .147 -> .236 (worse) |
| `SVGP_FAST_GPU` | svgp / taxi | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-svgp-gpu-taxi | svgp taxi 924 -> 903 | DROPPED-noise | -2.3% n=1; opt-in removed at merge |
| `XN_FAST_TILED_RBF` | - | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-ocsvm-tiled-istella | ocsvm istella 440 -> 445 | DROPPED-noise | +1.2% |
| `XN_PCS_SPARSE` | poly-count-sketch / taxi | lane/apple-fast-neighbors2 @ 5fb6edd3f | n2-pcs-sparse-taxi | poly-count-sketch taxi 0.3 -> 0.3 | DROPPED-noise | no change |
| (baseline, no switch) | - | lane/apple-fast-kapprox @ 10d5a7970 | kap-grp-base-taxi, kap-grp-base-istella | - | OPEN | baseline lines |
| `KERNEL_FAST_NYS_RR_EIGH` | nystroem / istella | lane/apple-fast-kernel @ 9e851777c | kernel-nys-rr-ist | - | OPEN | A/B queued, no judged result yet |
| `KPCA_RESIDENT` | kernel-pca / istella | lane/apple-fast-kapprox @ 10d5a7970 | kap-kpca-istella | kernel-pca istella 1,014 -> 294 | OPEN | digests differ; quality vs sklearn not yet checked |
| `XN_FAST_CLS2_OCSVM_RES / _2L / _CHUNK256` | ocsvm | lane/apple-fast-gap-cls2 @ 72602a339 | gapcls2-* | - | OPEN |  |

## Neural (95)

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
| `AFN_ATTN_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-all, afn-w2-lmgrad-fwd-attnall-forward | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-arena | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_FLASH` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-flash | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_FUSE_MLP` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-fuse-mlp, afn-w2-lmgrad-fwd-fusemlp-forward | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_FUSE_PRE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-fuse-pre | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_GQA_TILE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-gqa-tile | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_NORM_SG` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-norm-sg | - | OPEN | neural lines run after the classical queue |
| `AFN_ATTN_ROPE_CACHE` | neural | lane/apple-fast-neural @ 600237d7c | afn-attn-rope-cache | - | OPEN | neural lines run after the classical queue |
| `AFN_CNN_DIRECT` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-cnn-direct | - | OPEN | neural lines run after the classical queue |
| `AFN_EMB_ATOMIC_BWD` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-emb-atomic | - | OPEN | neural lines run after the classical queue |
| `AFN_EPI_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-m1-all, afn-w2-epi-m2-all, afn-w2-epi-m3-all, afn-w2-epi-samba-all-mamba-step | - | OPEN | neural lines run after the classical queue |
| `AFN_EPI_ALL + AFN_SAMBA_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-all-training-step | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM2_ALL + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-all, afn-w2-gemm2-gemm-bf16-all, afn-w2-gemm2-transformer-forward-all | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM2_BIGTILE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-bigtile, afn-w2-gemm2-gemm-bf16-bigtile | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM2_DBUF + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-dbuf, afn-w2-gemm2-gemm-bf16-dbuf | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM2_DIRECT_B + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-direct-b, afn-w2-gemm2-gemm-bf16-direct-b | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM2_SWIZZLE + AFN_GEMM_BF16_MMA + AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-gemm2-gemm-swizzle, afn-w2-gemm2-gemm-bf16-swizzle | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-all, afn-gemm-bf16-all, afn-gemm-int8-all | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_BF16_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-bf16-bf16mma | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_INT8_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-int8-int8mma | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_SIMDGROUP` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-simdgroup, afn-gemm-bf16-simdgroup | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_SIMDGROUP + AFN_GEMM_SPLITK + AFN_LM_WGRAD_SPLIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-wgrad-vs-gemmsplitk-train | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_SIMDGROUP + AFN_LMGRAD_ALL + AFN_LM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-all-on-stack-train | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_SPLITK` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-splitk | - | OPEN | neural lines run after the classical queue |
| `AFN_GEMM_TILESHAPE` | neural | lane/apple-fast-neural @ 600237d7c | afn-gemm-gemm-tileshape | - | OPEN | neural lines run after the classical queue |
| `AFN_LMGRAD_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-all-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-all-train, afn-lm-all-forward | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_BWD_EPILOGUE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-bwd-epilogue-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_BWD_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-bwd-fuse-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_BWD_NORM1_RESID` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-norm1-resid-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_BWD_NOSYNC` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-bwd-nosync-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_HEAD_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-head-fuse-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_NOSYNC` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-nosync-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_PARAM_VIEWS` | neural | lane/apple-fast-neural @ 600237d7c | afn-lm-param-views-train, afn-lm-param-views-forward | - | OPEN | neural lines run after the classical queue |
| `AFN_LM_WGRAD_SPLIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-lmgrad-wgrad-split-train | - | OPEN | neural lines run after the classical queue |
| `AFN_LOSS_FUSED` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-lossfused-mlp, afn-optim-lossfused-samba | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA1_CHUNKSCAN` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-chunkscan | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA1_FUSE_IN` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-fusein | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA2_SSD_MMA` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m2-ssdmma | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA3_BWD_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-m3-bwd-arena-step | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA3_BWD_CHUNK` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-m3-bwd-chunk-step | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA3_SISO_FUSED` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m3-sisofused | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-all, afn-mamba-m2-all, afn-mamba-m3-all | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-arena, afn-mamba-m2-arena, afn-mamba-m3-arena | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA_DEVICE_REFUSAL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mamba-m1-devrefusal, afn-mamba-m2-devrefusal, afn-mamba-m3-devrefusal | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA_PROJ_EPILOGUE` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-m1-proj, afn-w2-epi-m2-proj, afn-w2-epi-m3-proj (+2) | - | OPEN | neural lines run after the classical queue |
| `AFN_MAMBA_PROJ_EPILOGUE + AFN_MAMBA_PROJ_SPLITK` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-splitk-fwd, afn-w2-epi-samba-splitk-step | - | OPEN | neural lines run after the classical queue |
| `AFN_MLP_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-all | - | OPEN | neural lines run after the classical queue |
| `AFN_MLP_FUSED_STEP` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-fused | - | OPEN | neural lines run after the classical queue |
| `AFN_MLP_MULTISTEP` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-multistep | - | OPEN | neural lines run after the classical queue |
| `AFN_MLP_RESIDENT` | neural | lane/apple-fast-neural @ 600237d7c | afn-mlp-resident | - | OPEN | neural lines run after the classical queue |
| `AFN_OPTIM_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-all-mlp, afn-optim-all-samba, afn-optim-all-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_OPT_CLIP_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-clipfuse-mlp, afn-optim-clipfuse-samba, afn-optim-clipfuse-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_OPT_FUSE_SCAN` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-fusescan-mlp, afn-optim-fusescan-samba, afn-optim-fusescan-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_OPT_MULTITENSOR` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-multitensor-mlp, afn-optim-multitensor-samba, afn-optim-multitensor-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_OPT_RESIDENT_STATE` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-resident-mlp, afn-optim-resident-samba, afn-optim-resident-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_OPT_VEC4` | neural | lane/apple-fast-neural @ 600237d7c | afn-optim-vec4-mlp, afn-optim-vec4-samba, afn-optim-vec4-lm | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_ALL` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-all-training-fwd, afn-samba-all-training-step, afn-samba-all-mamba-step | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_ARENA` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-arena-fwd, afn-samba-arena-step | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_DEVICE_ADMIT` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-admit-fwd, afn-samba-admit-step | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_EMB_ATOMIC` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-emb-atomic-step | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_FUSE` | neural | lane/apple-fast-neural @ 600237d7c | afn-samba-fuse-fwd, afn-samba-fuse-step | - | OPEN | neural lines run after the classical queue |
| `AFN_SAMBA_FUSE + AFN_SAMBA_HEAD_GEMM` | neural | lane/apple-fast-neural @ 600237d7c | afn-w2-epi-samba-head-fwd, afn-w2-epi-samba-head-step | - | OPEN | neural lines run after the classical queue |
| `AF_FAST_NOFILL` | adafactor / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-nofill-adafactor | - | OPEN | neural lines run after the classical queue |
| `AF_FAST_RESIDENT` | adafactor / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-resident-adafactor | - | OPEN | neural lines run after the classical queue |
| `BPE_ALL` | bpe-encode / enwik8; bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-all, bpe-encode-all | - | OPEN | neural lines run after the classical queue |
| `BPE_ENCODE_DEVICE` | bpe-encode / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-encode-dev | - | OPEN | neural lines run after the classical queue |
| `BPE_ENCODE_DEVICE + BPE_LIVEBUF` | bpe-encode / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-encode-livebuf | - | OPEN | neural lines run after the classical queue |
| `BPE_GROUP_FILTER + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-filter | - | OPEN | neural lines run after the classical queue |
| `BPE_LIVEBUF + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-livebuf | - | OPEN | neural lines run after the classical queue |
| `BPE_MERGE_BATCH + BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-batch | - | OPEN | neural lines run after the classical queue |
| `BPE_TRAIN_DEVICE` | bpe-train / enwik8 | lane/apple-fast-bpe @ 3355d37b3 | bpe-train-dev | - | OPEN | neural lines run after the classical queue |
| `LN_FAST_NOFILL` | layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-nofill-layernorm | - | OPEN | neural lines run after the classical queue |
| `MOE_FAST_MMA (+ _KB32, _WIDE, _PF)` | moe / synthetic | lane/apple-fast-gap-misc @ 5e2eec7a3 | gapmisc-moe-* | - | OPEN | neural lines at the back of the queue |
| `OPT_FAST_MAP_DOWN` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-mapdown-adagrad, gapoptim-mapdown-rmsprop, gapoptim-mapdown-adamax, gapoptim-mapdown-nadam | - | OPEN | neural lines run after the classical queue |
| `OPT_FAST_PIPE_CH` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-ch2m-adagrad, gapoptim-ch2m-rmsprop, gapoptim-ch2m-adamax, gapoptim-ch2m-nadam | - | OPEN | neural lines run after the classical queue |
| `OPT_FAST_RAW_DOWN` | adagrad / synthetic; adamax / synthetic; nadam / synthetic; rmsprop / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-rawdown-adagrad, gapoptim-rawdown-rmsprop, gapoptim-rawdown-adamax, gapoptim-rawdown-nadam | - | OPEN | neural lines run after the classical queue |
| `SCHED_FAST_INLINE` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-inline-lrexp | - | OPEN | neural lines run after the classical queue |
| `SCHED_FAST_INLINE,SCHED_FAST_P64` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-inlinep64-lrexp, gapoptim-schedcheck | - | OPEN | neural lines run after the classical queue |
| `SCHED_FAST_P64` | - | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-p64-lrexp | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_LSTM_SCAN` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-scan-regsyn, gaplstm-scan-regtaxi, gaplstm-scan-clftaxi, gaplstm-scan-clfsyn | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-smem-regsyn, gaplstm-smem-regtaxi, gaplstm-smem-clftaxi, gaplstm-smem-clfsyn | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_LSTM_SCAN + SEQ_FAST_LSTM_SCAN_SMEM + SEQ_FAST_LSTM_WGRAD` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-all-regsyn, gaplstm-all-regtaxi, gaplstm-all-clftaxi, gaplstm-all-clfsyn | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_LSTM_WGRAD` | lstm-clf / synthetic; lstm-clf / taxi-hourly; lstm-reg / synthetic; lstm-reg / taxi-hourly | lane/apple-fast-gap-lstm @ 0d6cbc821 | gaplstm-wgrad-regsyn, gaplstm-wgrad-regtaxi, gaplstm-wgrad-clftaxi, gaplstm-wgrad-clfsyn | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_MAP_DOWN` | adafactor / synthetic; layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-mapdown-adafactor, gapoptim-mapdown-layernorm | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_PIPE_CH` | layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-ch2m-layernorm | - | OPEN | neural lines run after the classical queue |
| `SEQ_FAST_RAW_DOWN` | adafactor / synthetic; layernorm / synthetic | lane/apple-fast-gap-optim @ cf4513f8a | gapoptim-rawdown-adafactor, gapoptim-rawdown-layernorm | - | OPEN | neural lines run after the classical queue |

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
