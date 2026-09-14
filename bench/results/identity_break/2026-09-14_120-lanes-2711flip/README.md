# 120 lanes on four GPUs at the DEVIATION 2711 fix (2026-09-14, commit 83380ca6d)

The record that proves the split-K Gram flip (`core/gram_splitk.mojo` ships the scalar strided arm,
`-D MOJOLEARN_2711_GRAM_STRIDED_DEAD=1` restores the old one) moves no bit on the three vendors and
makes the RTX 5090 identical. Every column ran the 120-lane `tools/identity_break.py` (nine hostile
fixtures, two fits per cell, train, infer and model columns) at 83380ca6d, the host set built as the
CPU column beside the GPU set.

| column | box | cells |
|---|---|---|
| apple-m4 | Apple M4, this Mac, Metal, two-core cap, all 17 GPU bindings rebuilt at the flip commit | 1080: stable 1062, MOVED 9, refused 9 (the tokenizer lane; its host binding was not yet built in that tree; `apple-m4.tokenizer.json` is the same lane taken minutes later with it built, 9 stable) |
| nvidia-h100-sm_90a | RunPod H100 80GB HBM3 (`bench/results/e1g/2026-09-14_113530-nvidia-h100-identity-120-lanes-2711flip`) | 1080: stable 1071, MOVED 9 |
| amd-mi300x-gfx942 | Hot Aisle MI300X, 22.04 ROCm container (`...-amd-mi300x-hotaisle-identity-120-lanes-2711flip`) | 1080: stable 1071, MOVED 9 |
| nvidia-rtx5090-sm_120a | RunPod GeForce RTX 5090, Blackwell consumer (`...-nvidia-rtx5090-identity-120-lanes-2711flip`) | 1080: stable 1071, MOVED 9, refused 0 |

`diff.apple-nvidia-amd.txt`: `summary: IDENTICAL=1071, MOVED=9` and `summary (infer/model):
IDENTICAL=1476, N/A=684`. `diff.h100-mi300x-rtx5090.txt`: the same two lines, the 5090 as the
third column. `apple-m4.tokenizer.json` against the H100 and MI300X: 9 IDENTICAL x3. No DIVERGENT
cell anywhere. The nine MOVED cells are all `byte-lm-resident`, the lane hashing an object array
(fixed on main at 43f153247, after this commit); loss, params and logits of that lane are equal
between fits and to the stateless lane on every box. The lane fix and the harness's refusal of
object dtypes will make the next record clean, and that record replaces this one as the gate's
columns; until then the CPU identity gate diffs against the 47-lane record of the same morning.

What this record closes:

- DEVIATION 2711: the four cells that refused on the RTX 5090 at the old default (pca, tsvd, ols,
  ridge on the 17-column `odd` fixture, the Jacobi handed a Gram an sm_120a AOT pass had
  miscompiled) read IDENTICAL x3 with the H100 and the MI300X on train, infer and model; every other
  5090 cell is identical too. Four GPU architectures, three vendors, one answer on 1071 cells.
- The flip itself: the H100 and MI300X columns at the flip are identical on every one of the 1044
  cells the 118-lane run (arm 0) also has, and the Apple column matches both.
- mamba2-dtlimit: DIVERGENT between the H100 and the MI300X in the arm-0 run and MOVED once on the
  MI300X there; at this commit the two vendors agree on all nine fixtures. That is what a race on
  one side looks like across two runs, so the AMD side stays open (probe recipe in
  docs/lanes/BRIEF_resident_and_dtlimit_moved_2026-09-14.md); it is not caused or fixed by 2711.
