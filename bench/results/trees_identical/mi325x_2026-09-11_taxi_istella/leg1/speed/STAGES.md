## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg1/speed/baseline.gbdt-symmetric.taxi.r1000000.stage.log

lane gbdt-symmetric, shape taxi-1000000x16, round 550.189 ms, hash 8dbbd709c0290f8e, fits in log 2, per-tree tables 0 (-), loop table SymmetricTree

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 21.2 |
| entry | train_pre_quantize | 24.1 |
| entry | train_quantize_borders | 9.4 |
| entry | train_cindex_build | 5.7 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 0.7 |
| entry | train_fit_with_test | 470.9 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 512.2 |
| entry | gbdt_fit_host_copy_in | 15.1 |
| entry | gbdt_fit_train | 512.3 |
| entry | gbdt_fit_model_text | 1.6 |
| entry | gbdt_fit_total | 529.0 |
| loop | loop fit wall | 470.1 |
| loop | sym.hist | 108.5 |
| loop | sym.pstats | 18.4 |
| loop | sym.score | 22.9 |
| loop | sym.winner | 9.9 |
| loop | sym.split | 50.7 |
| loop | sym.drain | 3.0 |
| loop | est.move | 28.6 |
| loop | est.approx | 26.4 |
| loop | est.pstats | 28.7 |
| loop | est.readback | 24.1 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 148.9 |

## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg1/speed/baseline.gbdt-depthwise.taxi.r1000000.stage.log

lane gbdt-depthwise, shape taxi-1000000x16, round 924.348 ms, hash 007ce4cf646dd892, fits in log 2, per-tree tables 100 (depthwise), loop table Depthwise

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 27.9 |
| entry | train_pre_quantize | 23.6 |
| entry | train_quantize_borders | 9.2 |
| entry | train_cindex_build | 5.7 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 0.7 |
| entry | train_fit_with_test | 836.4 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 877.1 |
| entry | gbdt_fit_host_copy_in | 14.6 |
| entry | gbdt_fit_train | 877.1 |
| entry | gbdt_fit_model_text | 4.8 |
| entry | gbdt_fit_total | 896.4 |
| per-tree | fit wall summed over 100 trees | 274.5 |
| per-tree | hist.build | 77.5 |
| per-tree | split.chain | 50.2 |
| per-tree | hist.scan | 37.5 |
| per-tree | partstats | 19.4 |
| per-tree | split.host | 17.3 |
| per-tree | score.kernel | 13.9 |
| per-tree | hist.subtract | 12.4 |
| per-tree | hist.zero | 12.2 |
| per-tree | score.read | 11.3 |
| per-tree | split.sizes | 9.7 |
| per-tree | leaf.values | 3.4 |
| per-tree | model.build | 0.9 |
| per-tree | score.hostreduce | 0.9 |
| per-tree | host.plan | 0.7 |
| loop | loop fit wall | 835.6 |
| loop | est.move | 29.5 |
| loop | est.approx | 26.8 |
| loop | est.pstats | 29.5 |
| loop | est.readback | 26.9 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 448.4 |

## bench/results/trees_identical/mi325x_2026-09-11_taxi_istella/leg1/speed/baseline.gbdt-lossguide.taxi.r1000000.stage.log

lane gbdt-lossguide, shape taxi-1000000x16, round 1886.693 ms, hash 1330842b324c2fb5, fits in log 2, per-tree tables 100 (lossguide), loop table Lossguide

| where | stage | ms |
|---|---|---|
| Python | round minus gbdt_fit_total | 28.3 |
| entry | train_pre_quantize | 23.2 |
| entry | train_quantize_borders | 10.1 |
| entry | train_cindex_build | 6.5 |
| entry | train_targets_upload | 1.4 |
| entry | train_pre_fit | 0.8 |
| entry | train_fit_with_test | 1794.2 |
| entry | train_post_fit | 0.0 |
| entry | train_total | 1836.2 |
| entry | gbdt_fit_host_copy_in | 14.6 |
| entry | gbdt_fit_train | 1837.8 |
| entry | gbdt_fit_model_text | 6.0 |
| entry | gbdt_fit_total | 1858.4 |
| per-tree | fit wall summed over 100 trees | 1246.3 |
| per-tree | split.chain | 255.9 |
| per-tree | hist.build | 221.9 |
| per-tree | hist.scan | 176.2 |
| per-tree | split.host | 157.1 |
| per-tree | split.sizes | 85.7 |
| per-tree | partstats | 73.3 |
| per-tree | hist.subtract | 66.2 |
| per-tree | score.kernel | 61.9 |
| per-tree | hist.zero | 56.3 |
| per-tree | score.read | 51.1 |
| per-tree | host.plan | 6.2 |
| per-tree | leaf.values | 3.1 |
| per-tree | score.hostreduce | 2.0 |
| per-tree | model.build | 0.9 |
| loop | loop fit wall | 1793.4 |
| loop | est.move | 29.1 |
| loop | est.approx | 24.9 |
| loop | est.pstats | 27.7 |
| loop | est.readback | 25.0 |
| loop | loop wall minus per-tree walls minus est.* (unwrapped per-iteration work) | 440.4 |

