# tools/verify_external.sh run by an outsider (2026-09-14)

The recipe in docs/VERIFY_EXTERNALLY.md, run on a box that did not build the
wheel: a RunPod H100 (sm_90a) pod, image runpod/pytorch:2.4.0-py3.11-cuda12.4.1,
with pixi removed from PATH, the pod's own /usr/bin/python3 3.11.10, pip and
git, `mojolearn==0.8.5` from PyPI, a fresh public clone of this repository
(HEAD f7376c787 at the time), and the record `bench/results/identity_break/2026-09-14_47-lanes`
(commit 7bf4f4cc9, the CPU gate's GPU columns). Leg
`bench/results/e1g/2026-09-14_111755-nvidia` (untracked archive).

| what | value |
|---|---|
| box | RunPod NVIDIA H100 80GB HBM3 (sm_90a), image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 |
| python, numpy | /usr/bin/python3 3.11.10; numpy 2.4.4 (pip's newest that day; the repo's pixi env has 2.5.2) |
| wheel loaded | 0.8.5 (built from 8d16ce2f), vendor cuda, mode identical, from /usr/local/lib/python3.11/dist-packages |
| record, harness | 2026-09-14_47-lanes, harness at its commit 7bf4f4cc9 (newer than the wheel) |
| training cells | 423 (417 stable, 0 moved, 6 REFUSED: radius/base, radius/denormal, radius/denormal_ftz, radius/dupes, radius/odd, radius/negative, each `ValueError: setting an array element with a sequence.`) |
| diff, training | every non-refused cell IDENTICAL x4 (the three committed columns and this one) |
| diff, inference and model | IDENTICAL x4 on every cell (585 inference, 261 no-save) |
| script exit | 0 |

Findings, in the order an outsider meets them:

1. The script cannot run from a shipped source archive; it needs the
   repository clone (bench/results/ is where the record lives). The leg body
   cloned GitHub on the box; the clone is 2.9 GB.
2. `radius` REFUSED on six of nine fixtures (base, denormal, denormal_ftz,
   dupes, odd, negative) with `ValueError: setting an array element with a
   sequence.`; the outsider's numpy is 2.4.4 (pip's newest at the time), ours
   is 2.5.2, and the harness (7bf4f4cc9) is newer than the wheel (8d16ce2f).
   Open: harness/wheel skew or a numpy difference; the lane is owned by
   tools/identity_break.py.
3. The diff exited 0 with six REFUSED cells in the tested column, so the
   header's "exit 0 means every cell of your column equals the three
   committed columns" is not what the exit code measures; a positive count of
   IDENTICAL x4 rows (here 897 of 903) is.
4. pip had to install numpy; the recipe should say so, or the wheel should
   declare it.

The column file `nvidia-h100-sm_90a.json` is the outsider's; `verify_external.log`
is the script's complete output.

Fixed on main the same day (mojolearn-97): the script's verdict is positive
(no REFUSED or MOVED in the tested column, IDENTICAL > 0, no DIVERGENT, `VERDICT:
PASS` or `FAIL`), the header says run from a git clone, pip installs numpy with
the wheel, the harness's `_ragged` reads the wheel's ragged container by index
and length, and every column records its python and numpy versions under
`package`. **Rerun owed at a harness with the by-index `_ragged`**, as the
second row here; it is one command on the next H100 box.

## Second run (same day, 17:09Z), harness with the positive verdict

Same recipe on a fresh RunPod H100 pod, clone at 2b7f991b6 (the script with
`VERDICT: PASS|FAIL`, numpy installed with the wheel), the same 47-lane record
and wheel 0.8.5. Column `nvidia-h100-sm_90a.run2.json`, log `verify_external2.log`.

| what | value |
|---|---|
| training cells | 423 (417 stable, 0 moved, 6 REFUSED: the same six radius fixtures, same ValueError) |
| diff | IDENTICAL=423 over the stable cells, exit 0 |
| script verdict | `VERDICT: FAIL`, exit 1 (the six refused cells now fail the run, as they should) |

Why radius still refuses: the script runs the HARNESS AT THE RECORD'S COMMIT
(7bf4f4cc9, so the lanes are the record's), and the by-index `_ragged` fix
lives at c3e6dcd37 and later. A record taken at a commit that carries the fix
closes it; the next outsider run should use such a record, and a release
should ship the columns of a record taken at its own commit so the harness,
the wheel and the record agree (0.8.6 plan).
