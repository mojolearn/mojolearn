# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Train a byte-level BPE vocabulary from the command line.

    mojo run -I . tokenizer/train/train_main.mojo OUT_PREFIX VOCAB_SIZE \\
        MIN_FREQUENCY CORPUS_FILE [CORPUS_FILE ...]

writes two files, which are the two formats a trained vocabulary is written
in:

    OUT_PREFIX.ranks.tsv        OURS: `rank<TAB>hex`, what
                                `BpeTokenizer.from_ranks_file` and
                                `load_bpe_tokenizer_from` read
    OUT_PREFIX.tokenizer.json   the ecosystem's: what Hugging Face
                                `tokenizers` loads, so a model published with
                                this vocabulary is usable by people who do
                                not use mojolearn

EACH CORPUS FILE IS ONE DOCUMENT, and a pre-token never spans two of them, so
the ORDER the files are given in cannot reach the result. That is a property
of the trainer rather than a promise about this script
(`tools/bpe_trainer_determinism.py` measures it).

MOJOLEARN SHIPS NO CORPUS AND NO VOCABULARY. The corpus is yours.
"""

from std.sys import argv

from tokenizer.encoding import GPT2_ENDOFTEXT, GPT2_PAT_STR, string_bytes
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.train.bpe_train import train_bpe
from tokenizer.train.emit import (
    render_ranks,
    render_tokenizer_json,
    write_text,
)

comptime USAGE = (
    "usage: train_main.mojo OUT_PREFIX VOCAB_SIZE MIN_FREQUENCY CORPUS_FILE"
    " [CORPUS_FILE ...]"
)


def _read_bytes(path: String) raises -> List[UInt8]:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return string_bytes(text)


def main() raises:
    var args = argv()
    if len(args) < 5:
        print(USAGE)
        raise Error("train_main: too few arguments")

    var out_prefix = String(args[1])
    var vocab_size = Int(String(args[2]))
    var min_frequency = Int(String(args[3]))

    var documents = List[List[UInt8]]()
    var total = 0
    for i in range(4, len(args)):
        var doc = _read_bytes(String(args[i]))
        total += len(doc)
        documents.append(doc^)

    print(
        "train_main: "
        + String(len(documents))
        + " documents, "
        + String(total)
        + " bytes, vocab_size "
        + String(vocab_size)
        + ", min_frequency "
        + String(min_frequency)
    )

    var classes = builtin_unicode_classes()
    var vocab = train_bpe(documents, classes, vocab_size, min_frequency)

    var ranks_path = out_prefix + ".ranks.tsv"
    var json_path = out_prefix + ".tokenizer.json"
    write_text(ranks_path, render_ranks(vocab))
    write_text(
        json_path,
        render_tokenizer_json(
            vocab, String(GPT2_PAT_STR), String(GPT2_ENDOFTEXT)
        ),
    )

    print(
        "  "
        + String(vocab.n_tokens())
        + " tokens ("
        + String(vocab.n_merges())
        + " merges) from "
        + String(vocab.n_groups)
        + " pre-token groups, "
        + String(vocab.n_ties_broken)
        + " ties broken"
    )
    print("  wrote " + ranks_path)
    print("  wrote " + json_path)
