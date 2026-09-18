#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Train the decoder language model on a corpus: the corpus-to-steps entry point.

    python3 tools/lm_train.py --corpus training/corpus/enwik8/input.txt --out RUN   # trains OUR vocabulary
    python3 tools/lm_train.py --corpus ... --vocab ranks.tsv --out RUN               # YOUR token map
    python3 tools/lm_train.py --corpus ... --vocab encoder.json vocab.bpe --out RUN  # YOUR token map
    python3 tools/lm_train.py --corpus ... --bytes --out RUN                         # byte ids, vocab 256

WHY THIS FILE EXISTS. Before 2026-09-18 nothing in mojolearn took a corpus
to training steps. `LanguageModelTrainer.train_step(ids)` takes one
materialized `int32[B, L+1]` batch and fetches nothing; every loop that read a
corpus was a probe (`tools/lm_step_memory_probe.py`, `lm_recycle_probe.py`,
`lm_shards_probe.py`, `lm_shakedown_resume.py`), each over `CorpusBatches`,
which casts raw bytes to ids. This is the entry point that does it on
purpose, and tokenization is its default: the vocabulary is trained once (or
the caller's is loaded), the corpus is tokenized once and cached
(`mojolearn.lm_corpus.prepare`), and the model's `vocab_size` is the
vocabulary's `n_vocab`. `--bytes` keeps byte training by explicit choice, on
`CorpusBatches` exactly as the probes use it.

OUTPUT (`--out`, a fresh directory): `run.json` (the data schedule, including
the vocabulary's sha256 and n_vocab, the shape, the per-step losses), and
`state.npz` (`export_state()` arrays) unless `--no-state`. With
`--witness-rows` each step's embedding and unembedding gradients are reduced
per row and the rows at or above 256 are counted: under a vocabulary of more
than 256 ids the embedding rows of ids that occurred get nonzero gradient,
and under `--bytes` every embedding row at or above 256 must be EXACTLY zero.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: the 162M GPT-3 Small-shaped target of the probes, vocab replaced by the
#: vocabulary's n_vocab under a tokenized corpus
TARGET_SHAPE = [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
#: small enough for a one-step gradient witness at the full vocabulary
WITNESS_SHAPE = [1, 256, 64, 4, 2, 16, 128, 2, 50257]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _row_report(grad, vocab, name, ids):
    """Per-row reduction of an embedding-shaped gradient: rows are the
    vocabulary axis wherever it sits."""
    import numpy as np
    g = np.asarray(grad)
    axis = 0 if g.shape[0] == vocab else 1
    rows = np.abs(g).max(axis=1 - axis) if g.ndim == 2 else np.abs(g)
    live = rows != 0
    seen = np.unique(ids)
    seen_hi = seen[seen >= 256]
    return dict(
        tensor=name, shape=list(g.shape), vocab_axis=axis,
        rows_nonzero=int(live.sum()), rows_nonzero_below_256=int(live[:256].sum()),
        rows_nonzero_at_or_above_256=int(live[256:].sum()), rows_at_or_above_256=int(max(vocab - 256, 0)),
        max_abs_at_or_above_256=float(rows[256:].max()) if vocab > 256 else 0.0,
        distinct_ids_in_batch=int(seen.size), distinct_ids_at_or_above_256=int(seen_hi.size),
        seen_ids_at_or_above_256_with_nonzero_row=int(live[seen_hi].sum()) if seen_hi.size else 0,
        first_live_rows_at_or_above_256=[int(i) for i in np.nonzero(live[256:])[0][:8] + 256],
        sha256=_sha(np.ascontiguousarray(g).tobytes()),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a fresh output directory")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--vocab", nargs="+", default=None, metavar="FILE",
                      help="YOUR token map: a rank<TAB>hex file, or encoder.json vocab.bpe; nothing is trained")
    mode.add_argument("--bytes", action="store_true", help="byte ids (vocab 256 rows used), CorpusBatches")
    ap.add_argument("--cache", type=Path, default=None, help="lm_corpus cache (default $MOJOLEARN_LM_CACHE)")
    ap.add_argument("--vocab-size", type=int, default=None, help="ranks to train (default 50256)")
    ap.add_argument("--vocab-sample-bytes", type=int, default=None)
    ap.add_argument("--shape", nargs=9, type=int, default=None,
                    help="B L DM H KV HD FF LAYERS VOCAB; VOCAB is replaced by the vocabulary's n_vocab")
    ap.add_argument("--target", action="store_true", help="the 162M target shape")
    ap.add_argument("--witness-shape", action="store_true", help="the small full-vocabulary witness shape")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=93261)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--witness-rows", action="store_true")
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="prepare the tokenized corpus and stop")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=False)
    log = (args.out / "log.txt").open("a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    from mojolearn import lm_corpus
    record = dict(schema="mojolearn.lm-train.run.v1", argv=sys.argv[1:] if argv is None else list(argv))
    if args.bytes:
        from lm_step_memory_probe import CorpusBatches
        data_mode = "bytes"
    else:
        vocab = None
        if args.vocab:
            if len(args.vocab) == 1:
                vocab = args.vocab[0]
            elif len(args.vocab) == 2:
                vocab = tuple(args.vocab)
            else:
                ap.error("--vocab takes a rank file, or encoder.json and vocab.bpe")
        kw = {}
        if args.vocab_size is not None:
            kw["vocab_size"] = args.vocab_size
        if args.vocab_sample_bytes is not None:
            kw["vocab_sample_bytes"] = args.vocab_sample_bytes
        t0 = time.perf_counter()
        corpus = lm_corpus.prepare(args.corpus, vocab=vocab, cache_dir=args.cache, progress=say, **kw)
        record["prepare_seconds"] = time.perf_counter() - t0
        record["tokens_dir"] = str(corpus.tokens_dir)
        record["vocabulary_path"] = str(corpus.vocabulary_path)
        record["tokens_manifest"] = {k: v for k, v in corpus.manifest.items() if k != "source"}
        say(f"prepared {corpus!r} in {record['prepare_seconds']:.1f} s")
        data_mode = "tokens" if vocab is None else "tokens-user-vocabulary"
    record["data_mode"] = data_mode
    if args.prepare_only:
        (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        return 0

    import numpy as np
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    shape_args = list(args.shape or (TARGET_SHAPE if args.target else WITNESS_SHAPE if args.witness_shape
                                     else [1, 256, 64, 4, 2, 16, 128, 2, 50257]))
    if not args.bytes:
        shape_args[8] = corpus.n_vocab
    shape = Shape(*shape_args)
    if args.bytes:
        batches = CorpusBatches(args.corpus, shape.batch, shape.length)
        schedule = dict(fixture="lm_train bytes", corpus_sha256=batches.sha256, seed=args.seed,
                        schedule="CorpusBatches: step k row b: bytes[(k*batch*length + b*length) % "
                                 "(bytes - length - 1) : +length+1]")
    else:
        batches = corpus.batches(shape.batch, shape.length)
        schedule = batches.data_schedule(seed=args.seed)
    record["describe"] = batches.describe()
    record["shape"] = shape.to_dict()
    record["data_schedule"] = schedule

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if "norm" in entry["name"]:
            weights[entry["offset"]:entry["offset"] + entry["size"]] += np.float32(1)
    record["init_sha256"] = _sha(weights.tobytes())
    trainer = Trainer(weights, shape=shape, data_schedule=schedule, lr=args.lr, resident=args.resident,
                      step_result="full")
    steps = []
    for k in range(args.steps):
        ids = batches.ids(k)
        t0 = time.perf_counter()
        result = trainer.train_step(ids)
        seconds = time.perf_counter() - t0
        loss = float(np.asarray(result["loss"]).reshape(-1)[0]) if "loss" in result else None
        entry = dict(step=k, loss=loss, seconds=seconds, ids_sha256=_sha(np.ascontiguousarray(ids).tobytes()),
                     max_id=int(ids.max()), gradients_sha256=_sha(np.asarray(result["flat_gradients"]).tobytes()))
        if args.witness_rows:
            g = result["gradients"]
            entry["embed"] = _row_report(g["embed"], shape.vocab_size, "embed", ids[:, :-1])
            entry["lm_head"] = _row_report(g["lm_head"], shape.vocab_size, "lm_head", ids[:, 1:])
            say(f"step {k}: embed rows>=256 nonzero {entry['embed']['rows_nonzero_at_or_above_256']} of "
                f"{entry['embed']['rows_at_or_above_256']} (distinct ids>=256 in the batch "
                f"{entry['embed']['distinct_ids_at_or_above_256']}, max|g| {entry['embed']['max_abs_at_or_above_256']:.3e}); "
                f"lm_head rows>=256 nonzero {entry['lm_head']['rows_nonzero_at_or_above_256']}")
        steps.append(entry)
        say(f"step {k} loss {loss} {seconds:.3f} s max_id {entry['max_id']}")
    record["steps"] = steps
    record["run_data_schedule"] = trainer.data_schedule
    if not args.no_state:
        state = trainer.state_dict()
        np.savez(args.out / "state.npz", **{k: np.asarray(state[k]) for k in ("parameters", "m", "v", "flags")})
        record["state_sha256"] = {k: _sha(np.asarray(state[k]).tobytes()) for k in ("parameters", "m", "v", "flags")}
    (args.out / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    say(f"wrote {args.out / 'run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
