# SPDX-License-Identifier: Apache-2.0
"""The GPU tokenizer's native entry table (lane vdf2-tok, 2026-11-24).

The GPU tokenizer image (`_mojolearn_x_tokenizer`) publishes ONE entry
(`tokenizer_gpu_entry_v1` in its binding file, body
`tokenizer/gpu/corpus_gpu.mojo` `corpus_gpu_entry_dispatch`) at module init,
through `core/entry_table.mojo`, in the slot of its tier and vendor. The host
corpus door (`bindings/lm_corpus_public_native_door.mojo`) and the `models.Tokenizer` door
(`bindings/tokenizer_public_native_door.mojo`, connect_vocab) each resolve it ONCE at
import-time connection (`bindings/lm_corpus_metadata_door.mojo`
`corpus_gpu_connect`, reading the image's `x_tokenizer_numeric_mode` and
`x_tokenizer_vendor`), and the host RUN of `lm_corpus.prepare(device="gpu")`
calls it with the GIL released, over plain spans: no Python object crosses.

The entry never touches the GIL: its only caller is a RUN that already
released it.

THE ABI (`TOKENIZER_GPU_ENTRY_ABI`; addresses and ints are 64-bit words):

  TOK_GPU_OP_CORPUS_TRAIN   addresses [span, cuts, pairs_out, words_out]
                            ints [m, n_cuts, document_bytes, vocab_size, min_frequency, pairs_cap]
      The documents of the m bytes at `span` (cut at the n_cuts sorted int64
      offsets at `cuts`, 0 and m included, each piece into documents of at
      most document_bytes bytes ending at the last 0x0A: the host
      `_cut_documents`), trained on the device. The merges (int32 pairs, room
      `pairs_cap` words) and [n_merges, n_ties, n_groups] (int64) are written;
      the result is the document count.
  TOK_GPU_OP_CORPUS_ENCODE  addresses [data, points, ranks_text, ids_out, bounds_out, stats_out]
                            ints [n, n_points, document_bytes, ranks_bytes]
      The host `bpe_encode_corpus_into` on the device: the n bytes cut at the
      sorted int64 split points, each document encoded alone with the
      canonical rank text's tokenizer, ids (int32, room n) back to back, the
      token offset of every split point (int64) and [n_documents, max_id,
      ids_above_255] (int64). The result is the id count.

The `mojolearn.models.Tokenizer` device side (lane vdf2-tokmodel, tokenizer/gpu/vocab_gpu.mojo):

  TOK_GPU_OP_VOCAB_LOAD     addresses [rank_arena, rank_offsets, rank_lengths, rank_buckets, sp_arena,
                                       sp_offsets, sp_lengths, sp_ids, added_of]
                            ints [arena_bytes, n_ranks, n_buckets, mask, sp_arena_bytes, n_specials,
                                  n_added_of, pattern]
      The host `VocabTokenizer`'s tables (int64 words; the added tokens in the host's order)
      uploaded once. The result is the table's handle word (freed by TOK_GPU_OP_VOCAB_FREE).
  TOK_GPU_OP_VOCAB_ENCODE   addresses [handle, text, ids_out]   ints [n, allow_special]
      `VocabTokenizer.encode` of the n bytes at `text`: ids (int32, room max(n, 1)). The result is
      the id count.
  TOK_GPU_OP_VOCAB_DECODE   addresses [handle, ids, bytes_out]  ints [n_ids, cap]
      The bytes of the n_ids int64 ids (room cap). The result is the byte count, or -1 - p for the
      first id at position p that is neither an added token nor a rank.
  TOK_GPU_OP_VOCAB_FREE     addresses [handle]                  ints []
      The table released. The result is 0.

The `mojolearn.models.tokenizer.pretokenize(device="gpu")` side (lane vdl-pretok, 2026-11-24,
tokenizer/gpu/pretokenize_gpu.mojo):

  TOK_GPU_OP_VOCAB_CUT      addresses [text, bounds_out]        ints [n, pattern]
      The pre-token boundaries of the n bytes at `text` under the pattern code (0 gpt2, 1 llama3,
      2 qwen2; any other code is refused): the host `pretokenize_pattern`'s m + 1 offsets (int64,
      room n + 1), offset for offset. The result is the boundary count.

Publication and lookup are small metadata operations with no independent
numerical work to split. A layout change takes a new V suffix.
"""
from core.entry_table import EntryRef, entry_table_publish, entry_table_resolve
from core.native_descriptor_abi import NativeDescriptorCall

comptime TOKENIZER_GPU_ENTRY_ABI = "mojolearn.tokenizer.gpu.v3"
comptime TOK_GPU_OP_CORPUS_TRAIN = 0
comptime TOK_GPU_OP_CORPUS_ENCODE = 1
comptime TOK_GPU_OP_VOCAB_LOAD = 2
comptime TOK_GPU_OP_VOCAB_ENCODE = 3
comptime TOK_GPU_OP_VOCAB_DECODE = 4
comptime TOK_GPU_OP_VOCAB_FREE = 5
comptime TOK_GPU_OP_VOCAB_CUT = 6
comptime TOK_GPU_OP_COUNT = 7


def tokenizer_gpu_vendor_word(vendor: String) -> Int:
    """The key's vendor word of the image's `x_tokenizer_vendor` string: 1 metal, 2 cuda, 3 hip;
    -1 for anything else (a build with no GPU vendor publishes nothing)."""
    if vendor == "metal":
        return 1
    if vendor == "cuda":
        return 2
    if vendor == "hip":
        return 3
    return -1


def _publish[name: StaticString](mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    var schemas: List[String] = [String(TOKENIZER_GPU_ENTRY_ABI)]
    var operations: List[Int] = [TOK_GPU_OP_COUNT]
    var entries = List[NativeDescriptorCall]()
    entries.append(entry)
    entry_table_publish[name](mode, vendor, publisher, schemas, operations, entries)
    _ = schemas
    _ = operations
    _ = entries


def _resolve[name: StaticString](mode: Int, vendor: Int) raises -> EntryRef:
    return entry_table_resolve[name](mode, vendor, 0, String(TOKENIZER_GPU_ENTRY_ABI), TOK_GPU_OP_COUNT)


def tokenizer_gpu_publish(mode: Int, vendor: Int, publisher: Int, entry: NativeDescriptorCall) raises:
    """Publish the image's entry in its (tier, vendor) slot. A vendor word below 1 publishes
    nothing (its callers refuse with the table's "provider is not published")."""
    if vendor < 1:
        return
    if mode == 1 and vendor == 1:
        _publish["MojolearnEntriesTokenizerGpuIdenticalMetalV1"](mode, vendor, publisher, entry)
    elif mode == 0 and vendor == 1:
        _publish["MojolearnEntriesTokenizerGpuFastMetalV1"](mode, vendor, publisher, entry)
    elif mode == 1 and vendor == 2:
        _publish["MojolearnEntriesTokenizerGpuIdenticalCUDAV1"](mode, vendor, publisher, entry)
    elif mode == 0 and vendor == 2:
        _publish["MojolearnEntriesTokenizerGpuFastCUDAV1"](mode, vendor, publisher, entry)
    elif mode == 1 and vendor == 3:
        _publish["MojolearnEntriesTokenizerGpuIdenticalHIPV1"](mode, vendor, publisher, entry)
    elif mode == 0 and vendor == 3:
        _publish["MojolearnEntriesTokenizerGpuFastHIPV1"](mode, vendor, publisher, entry)
    else:
        raise Error("native tokenizer gpu entries: unsupported execution key")


def tokenizer_gpu_resolve(mode: Int, vendor: Int) raises -> EntryRef:
    """The published entry of the image keyed (tier, vendor), admitted once (not published, key
    differs and ABI differs refuse here, never at call time)."""
    if mode == 1 and vendor == 1:
        return _resolve["MojolearnEntriesTokenizerGpuIdenticalMetalV1"](mode, vendor)
    if mode == 0 and vendor == 1:
        return _resolve["MojolearnEntriesTokenizerGpuFastMetalV1"](mode, vendor)
    if mode == 1 and vendor == 2:
        return _resolve["MojolearnEntriesTokenizerGpuIdenticalCUDAV1"](mode, vendor)
    if mode == 0 and vendor == 2:
        return _resolve["MojolearnEntriesTokenizerGpuFastCUDAV1"](mode, vendor)
    if mode == 1 and vendor == 3:
        return _resolve["MojolearnEntriesTokenizerGpuIdenticalHIPV1"](mode, vendor)
    if mode == 0 and vendor == 3:
        return _resolve["MojolearnEntriesTokenizerGpuFastHIPV1"](mode, vendor)
    raise Error("native tokenizer gpu entries: unsupported execution key (tier " + String(mode) + ", vendor "
                + String(vendor) + ")")
