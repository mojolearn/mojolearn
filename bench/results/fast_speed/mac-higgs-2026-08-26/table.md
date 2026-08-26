# The FAST path against the vendor, one rented NVIDIA box

Parsed from `bench/results/fast_speed/mac-higgs-2026-08-26/logs`, 3 arm logs.

Device(s) reported by the arms themselves: Darwin_arm64

`ratio = median(ours) / median(theirs)`. **Above 1.0 means WE ARE SLOWER.**
The warm-up round is excluded from every statistic and printed on its own,
because torch pays an enormous first call and hiding that makes the median
read as the whole story. `min` is printed beside the median so a reader can
see when the box was busy: the further they are apart, the less the row is worth.

## Every arm, as measured

| lane | arm | shape | n | median ms | min ms | max ms | warmup ms | hash |
|---|---|---|---|---|---|---|---|---|
| et | ours | higgs-1000000x28 | 3 | 20944.393 | 20448.300 | 24367.175 | 19157.416 | fdce1a43cb984497 |
| et | sklearn-et-cpu | higgs-1000000x28 | 3 | 28761.257 | 28494.875 | 31483.933 | 30148.936 | MOVED(3) |
| gbdt-symmetric | catboost-cpu | higgs-1000000x28 | 3 | 3806.136 | 3464.231 | 4171.009 | 3643.295 | 8757eed035588bf8 |
| gbdt-symmetric | ours | higgs-1000000x28 | 3 | 3762.358 | 3710.756 | 3866.485 | 3977.090 | ac7cf7badff08c6b |
| rf | ours | higgs-1000000x28 | 3 | 26909.764 | 24814.080 | 29018.100 | 24327.078 | 3ffa2951595422d4 |
| rf | sklearn-rf-cpu | higgs-1000000x28 | 3 | 84216.725 | 69836.920 | 85412.066 | 64318.219 | MOVED(3) |

## Ours against each opponent

| lane | shape | opponent | ours ms | theirs ms | ratio ours/theirs | verdict |
|---|---|---|---|---|---|---|
| et | higgs-1000000x28 | sklearn-et-cpu | 20944.393 | 28761.257 | 0.73 | we are 1.37x FASTER |
| gbdt-symmetric | higgs-1000000x28 | catboost-cpu | 3762.358 | 3806.136 | 0.99 | we are 1.01x FASTER |
| rf | higgs-1000000x28 | sklearn-rf-cpu | 26909.764 | 84216.725 | 0.32 | we are 3.13x FASTER |

## Accuracy, because a faster learner that fits worse has not won

| lane | arm | metric | value |
|---|---|---|---|
| et | ours | logloss | 0.621923 |
| et | ours | auc | 0.762823 |
| et | sklearn-et-cpu | logloss | 0.621178 |
| et | sklearn-et-cpu | auc | 0.764018 |
| gbdt-symmetric | ours | logloss | 0.542247 |
| gbdt-symmetric | ours | auc | 0.800537 |
| gbdt-symmetric | catboost-cpu | logloss | 0.543225 |
| gbdt-symmetric | catboost-cpu | auc | 0.799673 |
| rf | ours | logloss | 0.538850 |
| rf | ours | auc | 0.809906 |
| rf | sklearn-rf-cpu | logloss | 0.538873 |
| rf | sklearn-rf-cpu | auc | 0.809778 |

## Scale UNKNOWN for every row in this run

No arm in these logs emitted `scale=` in its FSPEED header, so
this run is OLDER than the throughput/fixed-cost declaration and
the split below could not be made. The rows are listed under
FIXED-COST because that is the conservative bucket, but the label
is NOT a measurement here -- some of these lanes are genuinely
large (kmeans ships 4,000,000 x 32) and some are genuinely tiny
(krr ships 16 rows). Check each shape tag by hand before quoting
anything from this file, or re-run so the arms declare it.

## Rows, ranked -- scale UNDECLARED, check each shape by hand

Every row here is a lane whose fixture is small enough that both
arms are dominated by launch and dispatch latency, plus the Python
call overhead the vendor arm pays inside the clock and ours does
not. Read each as WHAT ONE FIT COSTS END TO END on this box.

A ratio here is not wrong, it is UNCLAIMABLE: it does not tell you
whose kernel is faster. Some of these lanes cannot be made bigger
for stated reasons -- `hdbscan`'s dense mutual-reachability arm
materializes an m x m matrix, and inventing a larger fixture to
make the number look like throughput would be inventing a dataset.
Others simply have no size knob yet, and that is owed work.

They are ranked and kept rather than deleted because the fixed
cost is a real thing a user pays on a small problem.

| rank | lane | shape | ours ms | best opponent | their ms | we are |
|---|---|---|---|---|---|---|
| 1 | gbdt-symmetric | higgs-1000000x28 | 3762.358 | catboost-cpu | 3806.136 | **1.01x FASTER** |
| 2 | et | higgs-1000000x28 | 20944.393 | sklearn-et-cpu | 28761.257 | **1.37x FASTER** |
| 3 | rf | higgs-1000000x28 | 26909.764 | sklearn-rf-cpu | 84216.725 | **3.13x FASTER** |

3 rows in total have an opponent: 0 throughput, 3 fixed-cost.

## Refused arms, kept rather than dropped

An arm that could not run is a result about this box and this image.
Deleting the row would make the table read as full coverage.

| lane | arm | reason |
|---|---|---|
| et | cuml-et-gpu | ModuleNotFoundError: No module named 'cuml' |
| et | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| et | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |
| rf | cuml-rf-gpu | ModuleNotFoundError: No module named 'cuml' |
| rf | lightgbm-cpu | ModuleNotFoundError: No module named 'lightgbm' |
| rf | lightgbm-cuda | ModuleNotFoundError: No module named 'lightgbm' |

## Determinism, reported and not judged

This is the FAST path. A hash that moves between rounds is EXPECTED here
and is recorded, not failed. It is the direct evidence for what the
IDENTICAL mode buys, measured on the same box in the same hour.

| lane | arm | shape | distinct hashes across rounds |
|---|---|---|---|
| et | sklearn-et-cpu | higgs-1000000x28 | 3 |
| rf | sklearn-rf-cpu | higgs-1000000x28 | 3 |

## Notes the arms printed

- `forest.et.r1000000.log`: lane=et arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.gbdt-symmetric.r1000000.log`: lane=gbdt-symmetric arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so
- `forest.rf.r1000000.log`: lane=rf arms=ours metric=devices delta=1.000000 reason=devices=cpu (auto); on a GPU vendor's box the vendors' CPU arms do not run, so a lane whose only opponent is a CPU library has NO legal opponent here and says so

