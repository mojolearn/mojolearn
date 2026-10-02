# Biggest GPU gaps, 0.8.34 boards (partial) + Sep 29 tree boards

Snapshot taken Oct 2, 01:35Z, while the boards were still running.

| Box | Races recorded |
|---|---|
| M3 Ultra | 147 / 431 |
| nvc1 L40S | 94 / 464 |
| Hot Aisle MI300X | 33 / 443 |

Each row compares ours IDENTICAL median against the fastest opponent median. Rows come from board.json via `../board-worst-rows-20261001/board_rows.py`. Tree rows are fit times parsed from the archived Sep 29 BOARD.md files with `tree_fit.py`, because trees run last on the board and none have run on 0.8.34 yet.

## Classical gaps (ratio = ours / opponent)

| Lane | Ratio | Notes |
|---|---|---|
| SGD family (sgd-reg, sgd-clf; NVIDIA istella) | 34x vs cuML | taxi 15x; the fit runs a host loop |
| sgd-ocsvm, perceptron, pa-clf, pa-reg | 5-12x | same family |
| isotonic | NVIDIA 17-19x, M3 12-16x vs sklearn | host route |
| knn-imputer taxi | M3 42x, NVIDIA 23x | |
| enet-cv / lasso-cv | M3 14-15x, NVIDIA 6x | |
| Apple dense linalg vs torch MPS | cholesky 9x, svd 7x, lu 5x | |

## Tree gaps

| Lane | Ratio | Opponent |
|---|---|---|
| gbdt-ordered taxi, NVIDIA | 11.8x | catboost-gpu |
| gbdt-rank-yetirank istella, AMD / M3 | 3.7x / 3.3x | lightgbm-cpu |
| gbdt-categorical taxi, AMD | 2.2x | |
| gbdt-lossguide istella, NVIDIA / M3 | 1.75x / 1.8x | |
| gbdt-depthwise taxi, M3 | 1.6x | |
