# docs/apple-fast/ab-neural: A/B request lines for the Apple FAST NEURAL tier

One file per afn lane (`<lane>.txt`), one line per comparison, plus `<lane>.md` explaining each
define in a paragraph (mechanism, expected effect, risk, board lanes touched) so the M3 manager
reads results without reading code. `tier.txt` holds the baselines (IDENTICAL vs FAST, no define)
for every GPU board lane.

## The line

    CMD lane/apple-fast-neural <tag> bash tools/afn_ab.sh <tag> <binding> <board-lane> <shape> 2 "" "-D MOJOLEARN_AFN_<DEFINE>"

- `binding`: linalg | transformer | mamba | training | byte_lm | embedding | x_cnn (bindings/build_<binding>.sh).
- `board-lane`: a GPU lane of tools/bench_board_neural.py LANES: lm-train-step, lm-forward, gemm, gemm-bf16,
  gemm-int8, transformer-forward, mamba1-forward, mamba2-forward, mamba3-forward, samba-train-step,
  samba-forward, mlp-train-step. The *-infer lanes and lm-host-train-step run our CPU and are refused.
- `shape`: full (the board shape) or small (a smoke).
- `2`: reps, alternations A B A B; AFN_ROUNDS (default 3) timed rounds per race.
- arm A `""` is FAST with no define (main's FAST kernels); arm B carries the define. The baseline form
  `identical fast` races the IDENTICAL build (`ours`) against the FAST build (`ours-fast`).
- One tag per board-lane x define, prefixed `afn-<lane>-`.

Results: `~/mq/out/race-<tag>/race.log`; `lq log m3 <tag> 'AFN-AB |AFN-DEF-SUMMARY|AFN-QUALITY|AFN-FAIL'`.
`AFN-DEF-SUMMARY <tag> A=<ms> B=<ms> ratio=<B/A>` (below 1.0: arm B's median is the lower one);
`AFN-QUALITY <tag> ... status=OK|DIFF|NONE` (the output judge, then the neural_fast_quality pair verdict
on the lanes it covers). tools/afn_ab.sh's header documents every line and env knob.

## Why this directory is NOT auto-queued

The M3 manager's watcher auto-queues `docs/apple-fast/ab/*.txt` (the classical and trees families)
as soon as a file lands. The neural lanes are written in parallel on seven branches
(lane/apple-fast-neural-<lane>) and merged into ONE branch, `lane/apple-fast-neural`, at the end;
every line here names that merged branch, and a binding built from a single lane's branch would
miss the other lanes' defines (and the byte LM FAST build comes from afn-lm). Nothing here may be
queued until the merge; the orchestrator queues these lines by hand (`lq add m3 CMD ...`) once
`lane/apple-fast-neural` exists and its FAST build (FBUILD) is green.
