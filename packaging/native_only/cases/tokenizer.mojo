# SPDX-License-Identifier: Apache-2.0
# Byte BPE training, deterministic vocabulary, encode/decode; no fixture files.
from tokenizer.encoding import string_bytes
from tokenizer.impl.bpe import bpe_append
from tokenizer.impl.pretokenize import pretokenize
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.train.bpe_train import train_bpe
from tokenizer.train.emit import render_ranks


def main() raises:
    var text = String("banana banana bandana banana bandana banana")
    var corpus = string_bytes(text)
    var documents = List[List[UInt8]]()
    documents.append(corpus.copy())
    var classes = builtin_unicode_classes()
    var vocab = train_bpe(documents, classes, 272, 2)
    var replay = train_bpe(documents, classes, 272, 2)
    if vocab.n_tokens() <= 256 or render_ranks(vocab) != render_ranks(replay):
        raise Error("native-only tokenizer: merges must occur and repeated training must agree")
    var table = RankTable()
    table._reserve(vocab.n_tokens())
    for token in range(vocab.n_tokens()):
        table.offset.append(len(table.arena))
        table.length.append(vocab.length[token])
        var at = vocab.offset[token]
        for j in range(vocab.length[token]):
            table.arena.append(vocab.arena[at + j])
        table._insert(token)
    var ids = List[Int]()
    var bounds = pretokenize(corpus, classes)
    for i in range(len(bounds) - 1):
        bpe_append(ids, table, corpus, bounds[i], bounds[i + 1])
    if len(ids) >= len(corpus):
        raise Error("native-only tokenizer: learned merges did not reduce token count")
    var decoded = List[UInt8]()
    for token in ids:
        table.append_token_bytes(decoded, token)
    if len(decoded) != len(corpus):
        raise Error("native-only tokenizer: round-trip length changed")
    for i in range(len(corpus)):
        if decoded[i] != corpus[i]:
            raise Error("native-only tokenizer: round-trip bytes changed")
    print("NATIVE_ONLY_PASS tokenizer")
