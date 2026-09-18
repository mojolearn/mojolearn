# 2026-09-18 bpe-builder-native: the Mojo trainer behind BpeVocabularyTrainer

Apple M4, CPU tokenizer host binding, one core, nice 19, `--repeats 2`, all nine fixtures,
lanes `tokenizer`, `bpe-trainer`, `bpe-vocabulary`, `tokenized-corpus`. `run_arm.sh` is the
exact command per arm (`<source tree> <host dir> <out.json> [env]`).

| file | arm | vs m4.before.json |
|---|---|---|
| m4.before.json | main 0e4715f7c (`git archive`), its own binding: BpeVocabularyTrainer = pure Python | - |
| m4.after.json | lane b77a26e93, its binding: BpeVocabularyTrainer = Mojo `bpe_train` | IDENTICAL=36 |
| m4.trainer.sabotage-build.json | lane binding built with -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (MOJOLEARN_HOST_ALLOW_SABOTAGE=1) | DIVERGENT=27 (every bpe-trainer, bpe-vocabulary, tokenized-corpus cell), IDENTICAL=9 (tokenizer: not reached, correct) |
| m4.trainer.sabotage-env.json | lane, clean binding, MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (reaches the Mojo trainer as `break_ties_high`) | DIVERGENT=27, IDENTICAL=9 (tokenizer) |

`base` hashes, before = after: tokenizer 08b10bbc6f4b565b, bpe-trainer 6ed8b49585df3d85,
bpe-vocabulary 875b2b4bc2a4c302, tokenized-corpus 2580bb7a34e5ae01 (the same values
2026-09-18_tokenized-corpus recorded). The build sabotage moves bpe-trainer/base to
f5172d25e6499662, the value the Python door's env sabotage produced in that record: the
reversed tie-break is the same vocabulary on either backend. The build arm is what shows the
after arm REACHED the Mojo trainer: the Python code is identical in both, only the binary
differs.
