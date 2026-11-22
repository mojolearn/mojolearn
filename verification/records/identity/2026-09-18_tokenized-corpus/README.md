# 2026-09-18 tokenized-corpus: the bpe-vocabulary and tokenized-corpus lanes

Apple M4, CPU host bindings, one core, nice 19, `--repeats 2`, all nine fixtures, lanes
`bpe-vocabulary` (tokenizer.TrainedBpeVocabulary) and `tokenized-corpus` (mojolearn.lm_corpus),
at commit 081e14fa9 (lane/tokenized-corpus). `run_arm.sh` is the exact command per arm.

| file | arm | vs m4.json |
|---|---|---|
| m4.json | clean | - |
| m4.replay.json | clean again | IDENTICAL=18 |
| m4.trainer.sabotage.json | MOJOLEARN_BPE_TRAINER_SABOTAGE=1 | DIVERGENT=18 |
| m4.host.sabotage.json | tokenizer host built with -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1 | DIVERGENT=18 |

The parts each arm moves are in the diff files and in each lane's docstring.
