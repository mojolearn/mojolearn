# 2026-09-18: a real vocabulary on the LM training path, witnessed on an RTX 4090

`tools/lm_vocab_witness_body.sh` via `tools/gemm_remote_leg.sh` (pod nqbmrf373yti99, RTX 4090
sm_89, AMD EPYC 7B13, 404-verified), source commit 59a640c36 (shipped by git archive); see `status.txt` for per-step exits. R2-staged
enwik8, pile_github and `vocab/mojolearn-bpe-50257-v1/ranks.tsv` (sha 3d547b17..., 50,256 ranks,
n_vocab 50,257).

GRADIENT WITNESS, shape b1 L256 d64 h4 kv2 hd16 ff128 layers2 vocab 50,257, same seed and the
same initial weights (init sha 4c517da13b1e16d9 in both), two steps each:

| arm | step | embed rows >= 256 nonzero | distinct ids >= 256 in batch | max abs embed grad >= 256 | lm_head rows >= 256 nonzero (max abs) |
|---|---|---|---|---|---|
| tokens (TokenBatches) | 0 | 57 of 50,001 | 57 | 2.826e-03 | 50,001 (2.906e-03) |
| tokens | 1 | 62 | 62 | 6.805e-03 | 50,001 (5.801e-03) |
| bytes (CorpusBatches) | 0 | 0 | 0 | 0.0 exactly | 50,001 (5.47e-07) |
| bytes | 1 | 0 | 0 | 0.0 exactly | 50,001 (8.52e-07) |

The live embedding rows are exactly the ids that occurred (rows_nonzero == distinct ids). The
UNEMBEDDING is not dead under bytes: softmax gives every row a small push-down gradient.

BYTE PATH vs MAIN (`tools/lm_step_memory_probe.py --corpus enwik8 --steps 2
--witness-every-step`, CONTROL shape): the branch's Python and a copy whose `_byte_lm_impl.py`
is main's exact bytes (sha d6948abc...) give EQUAL sha256 for loss, gradients, parameters, m, v
and flags at both steps; a seed-93262 arm DIFFERS on loss, gradients, parameters, m and v
(flags are zero in all arms).

TOKENIZE ONCE on this box: all of enwik8 -> 26,834,102 ids, sha256 2ad0c690...2c253, the SAME
id array the Apple M4 produced; 11.40 s of encode, 8.77 MB/s, 2.35 M ids/s on one core.
