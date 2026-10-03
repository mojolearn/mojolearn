# lane/apple-fast-bpe: byte-level BPE training and encode on the Apple GPU (FAST only)

FINDING FIRST: on main both board lanes run on the HOST CPU in every tier. The only tokenizer binding is
`_mojolearn_tokenizer_host` (build_host_family.sh refuses FAST), so FAST == IDENTICAL on bpe-train
(47.8 / 47.9 ms on the M3) is one binary measured twice. Detail and per-call profile:
docs/apple-fast/notes/bpe.md.

This lane adds a FAST-only GPU binding, `bindings/_mojolearn_tokenizer_fast.mojo` (build:
`MOJOLEARN_NUMERIC_MODE=fast bash bindings/build_tokenizer_fast.sh`, output
`python/mojolearn/_mojolearn_tokenizer_fast.so`, afc binding name `tokenizer_fast`), with the device
code in `tokenizer/fast/bpe_device.mojo`. `python/mojolearn/tokenizer.py` uses it only under
MOJOLEARN_NUMERIC_MODE=fast and only for the paths `bpe_fast_flags()` reports compiled; a build with no
define runs main's host route (that is arm A of every request line with `""`). The host
`_mojolearn_tokenizer_host.so` must be built in the tree too (pre-clock vocabulary training and every
path not routed). Every switch is `-D MOJOLEARN_BPE_<NAME>=1`, default OFF, compiled only under FAST +
an Apple GPU target. QUALITY BAR: the device paths must give the host's bytes exactly, so arm A and arm B
digests must be EQUAL on every line (same merges in the same order, same ids); a digest difference is a
bug, not noise.

Remaining host step in every device arm (stated, not hidden): pre-tokenization (and, for training, the
pre-token dedup) still runs on the host before the first upload.

## MOJOLEARN_BPE_TRAIN_DEVICE
Mechanism: the merge loop on the GPU. A device open-addressing pair table (int32 key `left * V + right`
+ 1, claimed by compare-exchange, an append-only list of occupied slots) holds every pair count. Each
pass is 3 launches with no wait: top-2 partial lists over the live entries (256 blocks), one threadgroup
merging the 256 lists and selecting the winner (highest count, then smallest key: the host's total
order, plus the tie count), and one thread per pre-token group rewriting its symbols left to right and
applying exact integer count deltas by atomics. One 32-byte state readback per 64 passes; one download of
the merge list at the end. Expected: 3,840 passes x 3 launches; the win depends on Apple's per-launch
enqueue cost (~20 us) against the host's ~48 ms, so this arm alone may be SLOWER; it is the base the
other train switches build on. Risk: atomics compare-exchange on Metal (`weak=True` in a retry loop),
the table bound (3 x symbols entries; overflow raises with code 2/3), the 64-pass overrun after done
(no-op launches).

## MOJOLEARN_BPE_MERGE_BATCH (implies TRAIN_DEVICE)
Mechanism: the selection keeps the top 8 and merges up to 7 pairs in one pass when it is EXACT: each later
pair shares no token with an accepted one, its count is strictly above the next list entry (hence above
every pair outside the list), and no accepted pair is a self pair (proof in the notes). Token-disjoint
pairs cannot overlap, so one rewrite applies them all. Expected: far fewer passes in the high-count
early merges (often several disjoint pairs well separated in count), so launches drop by up to ~7x there;
late merges with many ties fall back to one per pass. Risk: the K=8 shared lists (16 KiB per
threadgroup, fits-gated by comptime assert), a mistake in the exactness rule shows as a digest change.

## MOJOLEARN_BPE_GROUP_FILTER (implies TRAIN_DEVICE)
Mechanism: each group carries a 64-bit token-presence mask (bit = id & 63, set when a token appears,
never cleared, so a superset); the apply pass returns at once for a group whose mask lacks either token
of every selected pair, instead of scanning its symbols. Expected: less memory traffic per pass (most
groups hold neither token); helps most once ids spread over the 64 bits. Risk: none for exactness (the
mask can only over-include); small cost of 2 extra words per group.

## MOJOLEARN_BPE_ENCODE_DEVICE
Mechanism: encode_batch (allow_endoftext=False, the board's call) on the GPU: the host pre-tokenizes
each document alone (as now), then one thread per pre-token runs `bpe_append`'s min-rank merge loop
against a device copy of the rank table (the same FNV-1a hash and probe), and one thread per document
compacts its ids to the document's byte offset. One crossing, 2 launches, 1 wait; Python slices each
document's ids at its byte offset. Expected: the merge loop (most of the host's per-document work)
leaves the host threads; pre-tokenization remains. Risk: long pre-tokens are quadratic per thread (as on
the host); the upload of 4 MiB text plus the rank table every call.

## MOJOLEARN_BPE_LIVEBUF (implies TRAIN_DEVICE and ENCODE_DEVICE)
Mechanism: grow-only pooled device buffers (one u8, one i32, one i64) kept across calls, every array a
range inside them, so a warm call allocates nothing and holds 3 live buffers instead of ~12-15; under
encode the rank table is also kept on the device for the tokenizer's life (uploaded by its first
encode). Expected: fewer allocations and lower per-launch cost (Apple launch cost grows ~0.25 us per
live buffer); matters most for the 192 launches between readbacks in training. Risk: the pool is per
process and never shrinks (memory stays at the largest call's size).

## MOJOLEARN_BPE_ALL
Every switch above together (MERGE_BATCH + GROUP_FILTER + LIVEBUF + both device paths).

## Compile status (2026-10-03, lane stopped compiling on Andrew's order: slots jammed)

No Mojo build of this branch ran to completion here. `python3 -m py_compile python/mojolearn/tokenizer.py`
passed (rc=0); `tools/hooks/no_host_routes.py origin/main HEAD` reports no finding.

| build | result |
|---|---|
| FAST `-D MOJOLEARN_BPE_ALL=1` | compile owed: peer (was queued, killed before it got a slot) |
| FAST `-D MOJOLEARN_BPE_TRAIN_DEVICE=1` | compile owed: peer |
| FAST `-D MOJOLEARN_BPE_MERGE_BATCH=1` | compile owed: peer |
| FAST `-D MOJOLEARN_BPE_GROUP_FILTER=1` | compile owed: peer |
| FAST `-D MOJOLEARN_BPE_ENCODE_DEVICE=1` | compile owed: peer |
| FAST `-D MOJOLEARN_BPE_LIVEBUF=1` | compile owed: peer |
| FAST, no define | compile owed: peer |
| IDENTICAL | not applicable: no IDENTICAL-built file changed (the host binding and tokenizer/ sources are untouched; build_tokenizer_fast.sh refuses IDENTICAL) |

Risky compile sites, first build errors most likely here: `tokenizer/fast/bpe_device.mojo` `_pair_add`
(`Atomic.load` / `compare_exchange[weak=True]` / `fetch_add` on `unsafe_offset` pointers), the
`InlineArray[Int32, BPE_K]` lists in `bpe_topk_part_kernel` / `bpe_select_kernel` / `bpe_apply_kernel`,
`BpeMem.p8/p32/p64` (`DeviceBuffer.unsafe_ptr() + off` returned as `MutPointer[T, MutAnyOrigin]`),
`create_sub_buffer` uploads/downloads, `@fieldwise_init struct _Slot`, and the deferred-init
`t_arena`/`t_off`/`t_len`/`t_bk` across the `comptime if BPE_LIVEBUF` branches of `_encode_launch`.
