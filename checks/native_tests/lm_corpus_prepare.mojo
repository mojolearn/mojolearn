# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed RUN of corpus preparation (lane dw-neural-models, 2026-11-21):
`bindings/lm_corpus_prepare_run.mojo` over `bindings/lm_corpus_files_run.mojo`
and `core/native_json_value.mojo`. Host files only: the CPU column.

  0  the manifest writer's layout, byte for byte: two-space indent, `": "`,
     `{}` for an empty object, ASCII escapes with a surrogate pair above the
     BMP; the text parses back to a document that writes the same bytes; a
     repeated name keeps its place and takes the last value; `-0` is 0; a
     float token is kept as spelled; integer reads truncate toward zero.
  1  path text: normalisation keeps one leading slash or exactly two, drops
     empty components and `.`; join and parent are `rstrip("/") + "/" + leaf`
     and `rsplit("/", 1)[0]`.
  2  `prepare` on a small corpus in a private directory: the first call
     trains, encodes and writes; the second finds the same token directory
     and the same manifest bytes. Independent oracles: the token file's
     SHA-256 and length are the manifest's; the rank file's canonical text
     hashes to the vocabulary identity; the ids decode back to the corpus
     bytes; a caller's vocabulary with the same rank text lands on the same
     token directory.

The public argument admission, the returned records and the refusals stay in
the Python boundary test (test_lm_corpus_native_boundary.py).
"""
from std.os import getenv
from std.time import perf_counter_ns
from bindings.lm_corpus_files_run import (
    corpus_run_atomic_abort, corpus_run_join, corpus_run_mkdirs, corpus_run_normalize, corpus_run_parent,
    corpus_run_read, corpus_run_write_bytes,
)
from bindings.lm_corpus_prepare_run import (
    CorpusPrepareRequest, corpus_prepare_run, tokens_manifest_sha,
)
from core.native_json_value import (
    JsonDoc, json_add_array, json_add_bool, json_add_int, json_add_null, json_add_number, json_add_object,
    json_add_string, json_count, json_member, json_number_equals, json_number_int, json_parse, json_push, json_put,
    json_string, json_write,
)
from core.sha256 import sha256_hex, sha256_hex_text
from tokenizer.encoding import string_bytes
from tokenizer.impl.tokenizer_host_shell_impl import BpeHandle, bpe_decode_seq_bytes, bpe_handle_load
from tokenizer.impl.vocab_build import ranks_text
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases

comptime LM_PREPARE_FIXTURES = 3


def _same(left: List[UInt8], right: List[UInt8]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):  # small-loop(bytes of one small fixture file)
        if left[i] != right[i]:
            return False
    return True


def json_fixture() raises:
    var doc = JsonDoc()
    var root = json_add_object(doc)
    json_put(doc, root, "schema", json_add_string(doc, "s"))
    json_put(doc, root, "n", json_add_int(doc, 3))
    json_put(doc, root, "x", json_add_null(doc))
    json_put(doc, root, "t", json_add_bool(doc, True))
    var items = json_add_array(doc)
    json_push(doc, items, json_add_int(doc, 1))
    json_push(doc, items, json_add_int(doc, -2))
    json_put(doc, root, "list", items)
    json_put(doc, root, "empty", json_add_object(doc))
    json_put(doc, root, "text", json_add_string(doc, "a\"b\\c\né😀"))
    json_put(doc, root, "f", json_add_number(doc, "0.5"))
    var written = List[UInt8]()
    json_write(doc, root, written)
    var expected = string_bytes(
        "{\n  \"schema\": \"s\",\n  \"n\": 3,\n  \"x\": null,\n  \"t\": true,\n  \"list\": [\n    1,\n    -2\n  ],\n"
        + "  \"empty\": {},\n  \"text\": \"a\\\"b\\\\c\\n\\u00e9\\ud83d\\ude00\",\n  \"f\": 0.5\n}")
    require(_same(written, expected), "manifest layout differs from the writer's layout")
    var again = JsonDoc()
    var parsed = json_parse(again, written)
    var rewritten = List[UInt8]()
    json_write(again, parsed, rewritten)
    require(_same(rewritten, written), "a parsed manifest does not write back the same bytes")
    require(json_string(again, json_member(again, parsed, "text")) == "a\"b\\c\né😀", "string escapes do not round trip")

    var loose = JsonDoc()
    var record = json_parse(loose, string_bytes("{\"a\": 1, \"b\": -0, \"a\": 2, \"c\": 1.50, \"d\": 1e3, \"e\": 12.7}"))
    require(json_count(loose, record) == 5, "a repeated name must keep one member")
    require(json_number_int(loose, json_member(loose, record, "a")) == 2, "a repeated name takes the last value")
    require(json_number_int(loose, json_member(loose, record, "d")) == 1000, "1e3 reads as 1000")
    require(json_number_int(loose, json_member(loose, record, "e")) == 12, "12.7 truncates to 12")
    require(json_number_equals(loose, json_member(loose, record, "d"), 1000), "1e3 equals 1000")
    require(not json_number_equals(loose, json_member(loose, record, "e"), 12), "12.7 does not equal 12")
    var text = List[UInt8]()
    json_write(loose, record, text)
    require(_same(text, string_bytes("{\n  \"a\": 2,\n  \"b\": 0,\n  \"c\": 1.50,\n  \"d\": 1e3,\n  \"e\": 12.7\n}")),
            "member order, -0 or float spelling changed")


def path_fixture() raises:
    require(corpus_run_normalize("//a/./b//c/") == "//a/b/c", "two leading slashes are kept")
    require(corpus_run_normalize("///a") == "/a", "three leading slashes are one")
    require(corpus_run_normalize("") == ".", "an empty path is the current directory")
    require(corpus_run_normalize("a/../b") == "a/../b", "normalisation does not resolve ..")
    require(corpus_run_join("/a/b//", "c") == "/a/b/c", "join strips trailing slashes")
    require(corpus_run_parent("/a/b") == "/a", "parent of /a/b")
    require(corpus_run_parent("/a") == "", "parent of /a is the empty prefix")
    require(corpus_run_parent("a") == "a", "a path without a slash is its own prefix")


def prepare_fixture() raises:
    var root = corpus_run_join(String(getenv("TMPDIR", "/tmp")), "mojolearn-lm-prepare-" + String(Int(perf_counter_ns())))
    corpus_run_mkdirs(root)
    try:
        var text = String("")
        for i in range(160):  # small-loop(lines of the fixture corpus)
            text += "the quick brown fox " + String(i % 7) + " jumps over the lazy dog; pack my box with five dozen jugs\n"
        var corpus_bytes = string_bytes(text)
        var corpus = corpus_run_join(root, "corpus.txt")
        corpus_run_write_bytes(corpus, corpus_bytes)
        var request = CorpusPrepareRequest(corpus, corpus_run_join(root, "cache"), 300, 1, 1048576, 512)
        var first = corpus_prepare_run(request)
        require(not first.cached, "the first prepare must make the token directory")
        var second = corpus_prepare_run(request)
        require(second.cached, "the second prepare must find the token directory")
        require(second.tokens_dir == first.tokens_dir and second.ranks == first.ranks, "the cache key changed between calls")
        require(_same(first.manifest.raw, second.manifest.raw), "the manifest changed between calls")

        var tokens = corpus_run_read(corpus_run_join(first.tokens_dir, "tokens.i32"))
        require(sha256_hex(tokens) == tokens_manifest_sha(first.manifest), "the token file is not the one the manifest pins")
        require(json_number_equals(first.manifest.doc, json_member(first.manifest.doc, first.manifest.root, "bytes"), len(tokens)),
                "the manifest's byte count is not the token file's")
        require(json_number_equals(first.manifest.doc, json_member(first.manifest.doc, first.manifest.root, "tokens"), len(tokens) // 4),
                "the manifest's token count is not the token file's")

        var handle = BpeHandle()
        bpe_handle_load(handle, first.ranks)
        var rank_text = ranks_text(handle.tok.value().ranks)
        var vocabulary = json_member(first.manifest.doc, first.manifest.root, "vocabulary")
        require(json_string(first.manifest.doc, json_member(first.manifest.doc, vocabulary, "sha256")) == sha256_hex_text(rank_text),
                "the rank file does not hash to the vocabulary identity")
        require(json_number_equals(first.manifest.doc, json_member(first.manifest.doc, vocabulary, "n_vocab"), handle.tok.value().n_vocab()),
                "the vocabulary identity's size is not the rank file's")

        var ids = List[Int]()
        for i in range(len(tokens) // 4):  # small-loop(ids of the fixture corpus, little endian)
            ids.append(Int(tokens[4 * i]) | (Int(tokens[4 * i + 1]) << 8) | (Int(tokens[4 * i + 2]) << 16) | (Int(tokens[4 * i + 3]) << 24))
        require(_same(bpe_decode_seq_bytes(handle, ids, len(corpus_bytes) + 16), corpus_bytes), "the ids do not decode back to the corpus")

        var user = CorpusPrepareRequest(corpus, corpus_run_join(root, "cache"), 300, 1, 1048576, 512)
        user.user = True
        user.user_ranks = rank_text.copy()
        user.user_count = handle.tok.value().n_vocab() - 1
        user.user_source = String("<fixture>")
        var third = corpus_prepare_run(user)
        require(third.cached and third.tokens_dir == first.tokens_dir, "the same rank text must land on the same token directory")
    except error:
        corpus_run_atomic_abort(root)
        raise error^
    corpus_run_atomic_abort(root)


def run_fixture(unit: Int) raises:
    if unit == 0:
        json_fixture()
    elif unit == 1:
        path_fixture()
    else:
        prepare_fixture()


def test_cases() raises:
    check_cases("lm_corpus_prepare", LM_PREPARE_FIXTURES, run_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
