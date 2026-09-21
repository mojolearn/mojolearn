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

LONG RUNS AND CROSS-DEVICE REPLAY. A run that is to be compared with another
device does not need every array of every step. `--hash-every N` records the
sha256 of the complete training state (parameters, m, v, flags) after every
N completed steps and after the last one, in `run.json` and, line by line as
they are taken, in `state_hashes.jsonl`. `--checkpoint-every N` writes the
state as raw arrays plus `state.json` under `checkpoints/step_XXXXXXXX/` (the
`tools/lm_shakedown_resume.py` format, each array carrying its sha256).
`--resume DIR` restores such a checkpoint in a fresh process through
`load_state_dict` and continues at its completed step; it refuses a
checkpoint whose data schedule is not the one this corpus and seed produce,
because a resumed run that read different tokens would compare nothing.
`--record-window A:B` (repeatable, absolute step indices, B exclusive) takes
the full witness inside the window: sha256 of the loss, the gradient,
parameters, m, v and flags after each step. `--lean` (with `--resident`)
keeps the state on the device and returns only the loss outside the windows,
so a long run pays for the witness only where it is asked for. Batches are a
pure function of the absolute step index, so a resumed or replayed window
reads the same tokens as the original run. Two runs agree when their
`state_hashes` agree at every common step; a disagreement is localized by
re-running the stretch before it with a `--record-window` over it.
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


def _window(text):
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("a window is A:B, absolute step indices, B exclusive")
    a, b = int(a), int(b)
    if a < 0 or b <= a:
        raise argparse.ArgumentTypeError("a window needs 0 <= A < B")
    return a, b


def _state_hashes(state):
    import numpy as np
    return {k: _sha(np.ascontiguousarray(np.asarray(state[k])).tobytes()) for k in ("parameters", "m", "v", "flags")}


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
    ap.add_argument("--hash-every", type=int, default=0, metavar="N",
                    help="sha256 the complete state after every N completed steps and after the last")
    ap.add_argument("--checkpoint-every", type=int, default=0, metavar="N",
                    help="write checkpoints/step_XXXXXXXX/ after every N completed steps and after the last")
    ap.add_argument("--resume", type=Path, default=None, metavar="DIR",
                    help="restore a checkpoint directory and continue at its completed step")
    ap.add_argument("--record-window", type=_window, action="append", default=[], metavar="A:B",
                    help="full per-step witness for absolute steps A <= k < B (repeatable)")
    ap.add_argument("--lean", action="store_true",
                    help="with --resident: return only the loss outside the record windows")
    args = ap.parse_args(argv)
    if args.lean and not args.resident:
        ap.error("--lean requires --resident")
    if args.hash_every < 0 or args.checkpoint_every < 0:
        ap.error("--hash-every and --checkpoint-every take a positive step count")

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

    step_result = "lean" if args.lean else "full"
    if args.resume is not None:
        from lm_shakedown_resume import _load_state
        state = _load_state(args.resume, zero_moments=False)
        if json.dumps(state["data_schedule"], sort_keys=True, default=str) != json.dumps(schedule, sort_keys=True, default=str):
            raise SystemExit("--resume: the checkpoint's data schedule is not the one this corpus, shape and seed "
                             "produce; a resumed run must read the same tokens")
        cfg = state["config"]
        trainer = Trainer(state["parameters"], shape=shape, data_schedule=state["data_schedule"], lr=cfg["lr"],
                          betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"],
                          resident=args.resident, step_result=step_result)
        trainer.load_state_dict(state)
        first = int(state["completed_steps"])
        record["resumed_from"] = dict(directory=str(args.resume), completed_steps=first,
                                      state_sha256=_state_hashes(state))
        say(f"resumed {args.resume} at step {first}")
    else:
        rng = np.random.default_rng(args.seed)
        weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
        for entry in Trainer.parameter_registry(shape):
            if "norm" in entry["name"]:
                weights[entry["offset"]:entry["offset"] + entry["size"]] += np.float32(1)
        record["init_sha256"] = _sha(weights.tobytes())
        trainer = Trainer(weights, shape=shape, data_schedule=schedule, lr=args.lr, resident=args.resident,
                          step_result=step_result)
        first = 0
    record["first_step"] = first

    def in_window(k):
        return any(a <= k < b for a, b in args.record_window)

    def save_checkpoint(completed):
        from lm_shakedown_resume import _save_state
        directory = args.out / "checkpoints" / f"step_{completed:08d}"
        digests, at = _save_state(trainer, directory)
        if at != completed:
            raise SystemExit(f"checkpoint reports step {at}, the loop completed {completed}")
        record.setdefault("checkpoints", []).append(dict(completed_steps=completed, directory=str(directory),
                                                         state_sha256=digests))
        say(f"checkpoint at step {completed}: {directory}")

    hash_log = (args.out / "state_hashes.jsonl").open("a") if args.hash_every else None
    steps = []
    last = first + args.steps - 1
    for k in range(first, first + args.steps):
        ids = batches.ids(k)
        t0 = time.perf_counter()
        result = trainer.train_step(ids)
        seconds = time.perf_counter() - t0
        loss = float(np.asarray(result["loss"]).reshape(-1)[0]) if "loss" in result else None
        entry = dict(step=k, loss=loss, seconds=seconds, ids_sha256=_sha(np.ascontiguousarray(ids).tobytes()),
                     max_id=int(ids.max()))
        if not args.lean:
            entry["gradients_sha256"] = _sha(np.asarray(result["flat_gradients"]).tobytes())
        if in_window(k):
            if args.lean:
                gradients = trainer.export_gradients(named=False)["flat_gradients"]
            else:
                gradients = result["flat_gradients"]
            witness = _state_hashes(trainer.export_state())
            witness["gradients"] = _sha(np.ascontiguousarray(np.asarray(gradients)).tobytes())
            witness["loss"] = _sha(np.array([loss], np.float32).tobytes())
            entry["witness_sha256"] = witness
        completed = k + 1
        if args.hash_every and (completed % args.hash_every == 0 or k == last):
            line = dict(completed_steps=completed, state_sha256=_state_hashes(trainer.export_state()))
            record.setdefault("state_hashes", []).append(line)
            hash_log.write(json.dumps(line, sort_keys=True) + "\n")
            hash_log.flush()
        if args.checkpoint_every and (completed % args.checkpoint_every == 0 or k == last):
            save_checkpoint(completed)
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
