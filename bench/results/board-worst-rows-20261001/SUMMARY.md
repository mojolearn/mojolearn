# Worst board rows, 2026-10-01

Ratio = our IDENTICAL median / the fastest opponent's median, one row per race; made with `board_rows.py` (this dir).

- `archived-rows.tsv`: the archived boards the worst-row list was read from: L40S and MI325X `board-resume` (wheel 0.8.25,
  445/463 races) and M3 Ultra `board-resume-main-3d0da1acf` (wheel 0.8.31, 302 races), from ~/mojolearn-evidence/board-archive.
- `board-0833-rows.tsv`: the live 0.8.33 board roots (`~/board-0833` on each box) as of 17:10Z: L40S 97 races with an
  opponent, MI325X 65, M3 Ultra 95. The qr, adafactor and lr-* lanes have not run on 0.8.33 yet.

Rows that moved most from the archived boards to 0.8.33 (same opponent medians, opponents measured once):
connected-components 167-226x -> 11-12x; lars/lasso-lars istella MI325X 141x -> 12x/3.8x; sgd-clf L40S istella 125x -> 37x;
perceptron / pa-clf / sgd-ocsvm MI325X taxi 59-98x -> 1.4-2.8x; skewed-chi2 MI325X 71x -> (not yet run).
Still large on 0.8.33: knn-imputer L40S istella 102x, M3 taxi 45x; poisson MI325X taxi 56x, L40S 13x; isotonic 16-42x everywhere; sgd-clf L40S 12-37x.
