# Lane afn-tier notes: the Apple FAST neural tier plumbing (2026-10-03)

Branch `lane/apple-fast-neural-tier` (worktree ~/mojolearn-wt/afn-tier), cut from origin/main 8897404da
(0.8.35). No kernel changed; nothing was run (the tier lane never measures): py_compile and bash -n only.

## What the tier lane changed

| file | change |
|---|---|
| tools/bench_board.py | `check_neural_modes(families, modes, vendor)`: a neural plan without identical is refused on NVIDIA and AMD only; `our_arms("neural", modes)` follows the modes (`ours` identical, `ours-fast` fast), so an Apple board (modes_for gives fast,identical) plans both arms on the 12 GPU neural races; race `modes`, `race_settings["numeric_mode"]`, the plan line (`neural: ours IDENTICAL + FAST (ours-fast, the Apple tier)`) and the not-covered lines say so. Docstring blocks updated. CPU lanes (`*-infer`, lm-host-train-step) stay off the plan in every mode. |
| tools/bench_board_neural.py | `OUR_ARMS = ("ours", "ours-fast")`, `OUR_MODE`; `_worker_env` sets MOJOLEARN_NUMERIC_MODE=fast for `ours-fast` exactly as classical_two_datasets/bench_board_more do for their `ours-fast`; the worker refuses a mode other than its arm's, a binary reading back another tier, and (for fast) a vendor other than apple; LM mode readback maps native_numeric_mode 0/1/2 to fast/identical/deterministic; `race --keep-outputs` keeps each arm's outputs npz at `<out>/<lane>-<data>-<arm>.outputs.npz` for the A/B judge; CPU lanes refuse every `ours*` arm by name. Quality of `ours-fast` is computed against `ours` like an opponent's (loss diffs, max abs/rel output diff) when both race. |
| tools/neural_fast_quality.py | `--package-dir` / `MOJOLEARN_NFQ_PACKAGE_DIR` picks the mojolearn/ a judge runs (afn_ab.sh installs arm A then arm B into the tree's package); `compare --min-seeds` (default 1, Andrew's one-seed A/B default; the 5-seed rule text stays); `pair --ref A.json --cand B.json --tag T` judges two runs of samba, mlp or blocks from the same seed and batches and prints `AFN-QUALITY`. Records carry `package_dir`. |
| tools/afn_ab.sh (new) | the A/B tool, interface and lines in its header and below. |
| tools/afn_ab.py (new) | its helper: `summary` (AFN-AB per arm + AFN-DEF-SUMMARY from the AFN-AB-RUN lines) and `compare-outputs` (the output judge). |
| tools/test_bench_board.py | the plan counts and the refusal test follow the tier (apple 328 cells, neural 64; FAST-only neural on Apple plans `ours-fast`; NVIDIA/AMD refused by name). Not in the ownership list; no afn lane edits it and a stale test would fail on merge. |
| docs/apple-fast/PLAN-neural.md, ab-neural/README.md, ab-neural/tier.txt, ab-neural/tier.md | the plan, the request-line form, the baselines. |
| CONTRIBUTING.md (Numeric modes paragraph), SUPPORT_MATRIX.md (neural FAST row) | the tier statement. |

## afn_ab.sh as implemented

    afn_ab.sh <tag> <binding> <board-lane> <shape> <reps> "<defines A>" "<defines B>"
    afn_ab.sh <tag> <binding> <board-lane> <shape> <reps> identical fast        (baseline)

1. Checks every argument (usage on any error). 2. Builds arm A then arm B: `MOJOLEARN_NUMERIC_MODE=<mode>
MOJOLEARN_MOJO_BUILD_FLAGS="<defines>" MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build_<binding>.sh`
(fast unless the arm word is `identical`); byte_lm builds into a fresh `MOJOLEARN_BYTE_LM_OUTDIR`
because its script refuses an existing output; each .so is copied to `~/afn-def/<tag>/<A|B>.so`
(AFN_SKIP_BUILD=1 reuses; a failed build prints the first errors and exits 1). 3. For rep 1..reps:
install A's .so into the package (`python/mojolearn/` for fast, `python/mojolearn/identical/` for
identical; AFN_FAST_DIR / AFN_IDENTICAL_DIR), race it, install B, race it. A race is
`MOJOLEARN_NUMERIC_MODE=<mode> MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=<repo>/python <python>
tools/bench_board_neural.py race --lane L --shape S --arms ours|ours-fast --rounds AFN_ROUNDS --out
~/mq/out/race-<tag>/<A|B>-<rep>/res --work ... --keep-outputs`, its NEURAL line parsed into
`AFN-AB-RUN`. 4. `afn_ab.py summary` prints `AFN-AB <tag> arm=A|B lane= shape= median_ms= rounds= ...
defines=''` per arm and `AFN-DEF-SUMMARY <tag> A= B= ratio=B/A`. 5. Judge: `afn_ab.py compare-outputs`
on the last rep's kept outputs (train lanes: step losses, rel 1e-3; forward lanes: max abs and rel
output diff, rel 1e-4 of max|A|; `AFN-QUALITY ... status=OK|DIFF|NONE`), then neural_fast_quality
for arm A then arm B (samba lanes: `samba --seed 0 --steps 100` with `--init-out`/`--init`, corpus
from training/corpus/; mlp-train-step: `mlp-data` + `mlp --dataset wine --seed 0`; the block forwards:
`blocks --seeds 1`), judged by `pair`; gemm and lm lanes have no subcommand and use the output judge
only (said on an AFN-QUALITY-TOOL SKIPPED line). 6. Arm B's .so is left installed.

Interpreter: AFN_PYTHON, else the newest `~/board-*/cache/venv/bin/python`, else python3 (numpy is
needed; torch is not, no opponent runs). Output: `~/mq/out/race-<tag>/` (AFN_OUT). Assumptions I could
not check: (a) a FAST byte LM .so installs at `python/mojolearn/_mojolearn_byte_lm.so` (afn-lm's
build; override with AFN_FAST_DIR if it lands elsewhere); (b) the tree is built (FBUILD) so the FAST
imports a binding needs exist; (c) `training/corpus/<name>/input.txt` is staged on the M3 for the samba
judge, else that judge is SKIPPED by name and the output judge stands.

## How the board shows the tier

An Apple board run (`--vendor apple`, default modes fast,identical) now plans every GPU neural race
with `ours` (IDENTICAL) and `ours-fast` (FAST) beside the torch arms: 12 races, 64 cells. The rendered
table's `ours FAST / arm` column (already there for trees and classical) fills for neural; the
`ours-fast` row's quality columns are its differences from `ours`. NVIDIA and AMD plan `ours` only.
