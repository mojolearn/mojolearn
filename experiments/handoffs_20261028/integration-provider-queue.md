# Integrator: provider migration READY queue -> main (2026-10-28)

Worktree ~/mojolearn-wt/integration-provider-queue-20261028, branch integration/provider-queue-20261028 from main 4771b55daa.

## Merged / picked (in order)
1. merge aefc4545aa (peer1 descriptors + IVF/Radius/PCS; includes 33a63ba6ad; NOT tip 4ad607eff5) - clean
2. merge 0456ee3e6a cluster refit lifetime - clean
3. merge 0f0b336d87 archive label/kernel (contains 68f5140002 + 5e7c530779) - clean
4. merge a3a8150111 linalg helper - clean
5. cherry-pick -x 32276a868c cca559b8a0 6be526212e 74c1e58ebe (metrics/resample) - merge of 74c1 conflicted only because lane base carries pre-cherry-pick duplicates already in main; picks clean (DECISIONS.txt)
6. cherry-pick -x f160cd1283 e456212ca7 4b9999bee4 (ARIMA/VAR/STL/companion; pair kept together) - one import-block union in bindings/tsa_public_native.mojo (DECISIONS.txt)
7. merge 654531acce Program owned distributed - clean
8. merge 7cacec39ba SVM state/coef - clean
9. merge 8e67e1bce1 peer4 Array.from_buffer (+ fcee3d4bfc ff81abe394) - clean
10. cherry-pick -x c7694b3277 9a0f043190 36a66a6885 2f3e60c2ca (peer2 batch8; e8a10/808a23/9608 excluded) - clean
11. merge 77ed135648 GBDTAdapter params - clean
HOLD not included (verified not ancestors): b00f153308, 9608dee4e5, 808a23e786, e8a10e89c0, 4ad607eff5, peer3-*, peer4-mode-boundaries.

## Compile
Roots: receipt_entries.txt (24 receipt roots + peer2 umap_init root from 2f3e tree) + classical_host + tokenizer_host (uncovered changed modules per closure_uncovered.txt) + device x_sequence/solver at sm_89 and gfx942.

## Tranche 2 (queued by coordinator; after tranche 1 is on main)
5b79a1b449 (peer4 mode), 2c746e42cd (peer2 prep, incl 9608), 213e36ec71 (peer2 PCA; check main's batch6 PCA not lost), d950eac50c (Prophet pair), 65d659a023 (timeseries pandas; supersedes 4b99), 20089632e6 (metrics fields; supersedes 74c1), a1cb276bf9 (resample pandas dtypes).
Extra roots: public_attribute_c, full_reset_params_c, native_mode_recipe, prep_stateless_c, pca_c, xseq/TSA dispatchers, metrics_program_fields, resample reader.
