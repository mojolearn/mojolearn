## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg2/speed/baseline.gbdt-symmetric.istella.r1000000.stage.log

lane gbdt-symmetric, shape istella-1000000x220, round 1559.02 ms, hash ae81b94e057b63a3, fits in log 2, per-tree tables 0 (-), loop table SymmetricTree

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 337.2 |
| entry | train_pre_quantize | 186.9 |
| entry | train_quantize_borders | 94.3 |
| entry | train_cindex_build | 35.5 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 6.7 |
| entry | train_fit_with_test | 699.2 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 1024.1 |
| entry | gbdt_fit_host_copy_in | 191.2 |
| entry | gbdt_fit_train | 1024.1 |
| entry | gbdt_fit_model_text | 6.5 |
| entry | gbdt_fit_total | 1221.8 |
| loop | loop fit wall | 698.3 |
| loop | sym.hist | 328.6 |
| loop | sym.pstats | 18.3 |
| loop | sym.score | 28.8 |
| loop | sym.winner | 12.3 |
| loop | sym.split | 49.4 |
| loop | sym.drain | 3.1 |
| loop | est.move | 28.7 |
| loop | est.approx | 26.4 |
| loop | est.pstats | 27.2 |
| loop | est.readback | 25.1 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 150.6 |

## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg2/speed/baseline.gbdt-depthwise.istella.r1000000.stage.log

lane gbdt-depthwise, shape istella-1000000x220, round 2135.261 ms, hash 79627a002ef5914f, fits in log 2, per-tree tables 100 (depthwise), loop table Depthwise

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 353.8 |
| entry | train_pre_quantize | 184.6 |
| entry | train_quantize_borders | 86.1 |
| entry | train_cindex_build | 37.8 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 6.7 |
| entry | train_fit_with_test | 1250.8 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 1567.5 |
| entry | gbdt_fit_host_copy_in | 190.8 |
| entry | gbdt_fit_train | 1581.2 |
| entry | gbdt_fit_model_text | 9.5 |
| entry | gbdt_fit_total | 1781.5 |
| per-tree | fit wall summed over 100 trees | 496.0 |
| per-tree | hist.build | 281.3 |
| per-tree | hist.scan | 52.6 |
| per-tree | split.chain | 48.4 |
| per-tree | partstats | 19.4 |
| per-tree | split.host | 16.6 |
| per-tree | score.kernel | 15.6 |
| per-tree | hist.subtract | 13.0 |
| per-tree | hist.zero | 12.4 |
| per-tree | score.read | 11.5 |
| per-tree | split.sizes | 10.0 |
| per-tree | leaf.values | 3.4 |
| per-tree | score.hostreduce | 3.0 |
| per-tree | model.build | 0.8 |
| per-tree | host.plan | 0.7 |
| loop | loop fit wall | 1250.0 |
| loop | est.move | 29.4 |
| loop | est.approx | 27.5 |
| loop | est.pstats | 29.0 |
| loop | est.readback | 26.6 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 641.4 |

## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg2/speed/baseline.gbdt-lossguide.istella.r1000000.stage.log

lane gbdt-lossguide, shape istella-1000000x220, round 3203.345 ms, hash 3f655048fb510d2a, fits in log 2, per-tree tables 100 (lossguide), loop table Lossguide

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 356.4 |
| entry | train_pre_quantize | 184.4 |
| entry | train_quantize_borders | 89.8 |
| entry | train_cindex_build | 36.7 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 6.7 |
| entry | train_fit_with_test | 2313.8 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 2632.9 |
| entry | gbdt_fit_host_copy_in | 190.0 |
| entry | gbdt_fit_train | 2645.5 |
| entry | gbdt_fit_model_text | 11.4 |
| entry | gbdt_fit_total | 2846.9 |
| per-tree | fit wall summed over 100 trees | 1566.2 |
| per-tree | hist.build | 404.9 |
| per-tree | split.chain | 279.4 |
| per-tree | hist.scan | 247.9 |
| per-tree | split.host | 159.4 |
| per-tree | split.sizes | 95.4 |
| per-tree | partstats | 75.4 |
| per-tree | hist.subtract | 71.1 |
| per-tree | score.kernel | 71.0 |
| per-tree | hist.zero | 60.9 |
| per-tree | score.read | 56.5 |
| per-tree | host.plan | 6.9 |
| per-tree | score.hostreduce | 5.4 |
| per-tree | leaf.values | 3.0 |
| per-tree | model.build | 0.9 |
| loop | loop fit wall | 2313.0 |
| loop | est.move | 29.5 |
| loop | est.approx | 26.5 |
| loop | est.pstats | 26.5 |
| loop | est.readback | 26.3 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 637.9 |

