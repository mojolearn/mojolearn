#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
r"""
identity_trace_diff.py -- cross-GPU identity-trace differ.

WHAT THIS IS FOR
================
A mojolearn GBDT fit can emit an "identity trace": an ordered list of stage
checkpoints, each one a stable stage name (tag) plus an FNV-1a64 hash over the
raw little-endian bit patterns of some buffer at that point in the fit. Run the
same fit on an Apple GPU and on an NVIDIA GPU, dump one trace per side, and
feed both to this tool. It answers the question that "the final models differ"
cannot answer: WHICH STAGE FIRST DIVERGED.

If raw `.bin` dumps sit beside the trace files, the tool goes one level deeper
on the first diverging record and reports per-cell bit patterns, decoded float
values, signed ULP distance, and a per-cell classification. The classification
exists to separate diagnoses that look alike in a hash but are not alike at
all. In particular DENORMAL-vs-ZERO is called out by name: Metal flushes
subnormals to zero, CUDA does not by default, and that single difference will
paint an entire buffer as "different" while being a one-line fix. A scattered
1-ULP divergence (reassociation, FMA contraction, a different reduction order)
is a completely different investigation from an all-DENORMAL-vs-ZERO one, and
the SUMMARY block is there so you can tell them apart in one glance.

WHAT THIS TOOL CANNOT TELL YOU
==============================
1. A MATCHING HASH IS NOT PROOF THE COMPUTATION WAS IDENTICAL. It proves the
   two buffers held the same bits at that checkpoint. Two different code paths
   can land on the same bits; a buffer can be right for the wrong reason;
   anything not hashed is invisible. Absence of divergence at a checkpoint is
   evidence about the buffer, not about the algorithm.
2. AN UNMATCHED TAG SET IS USUALLY A BIGGER FINDING THAN ANY HASH. If the two
   sides do not emit the same stages, the two runs took different code paths.
   Comparing hashes across differing stage sets is not meaningful on its own:
   you are no longer comparing the same computation. Fix the path difference
   first, then re-run this tool.
3. The tool cannot tell you WHY a stage diverged. It localizes, it does not
   explain. It also cannot tell you that an earlier, unhashed stage was fine.
4. It cannot distinguish "the writer emitted a wrong hash" from "the buffer
   really differs" -- except where a `.bin` dump is present, in which case it
   re-hashes the dump and verifies. A dump that does not hash to its recorded
   hash means the writer and this reader disagree about the format, and EVERY
   other conclusion in the report is void; that case exits 2.
5. It does not interpret integer divergence beyond "differs". Bit-exactness is
   the only meaningful relation for u8/u32/i32/i64 buffers here.

FILE FORMAT
===========
UTF-8 text, one record per line, TAB separated, exactly five fields:

    <seq>\t<tag>\t<dtype>\t<count>\t<hash>

  seq    decimal integer, 0-based, strictly increasing by 1 within a file
  tag    stable stage name, e.g. "tree03.level02.hist.after_scan"; ASCII, no tabs
  dtype  one of: f32 u32 i32 u8 f64 i64
  count  decimal integer, number of ELEMENTS hashed
  hash   exactly 16 lowercase hex digits, FNV-1a64 over the little-endian bytes
         of the elements in index order

Lines beginning with '#' are comments and are skipped. Blank lines are skipped.
A malformed line is a hard error naming the file and the line number.

Optional raw dumps sit beside the trace as
    <tracepath>.<seq>.<sanitized-tag>.bin
holding the little-endian element bytes for that record.

EXIT CODES
==========
  0  the traces are identical (same tag sequence, same counts, same hashes)
  1  the traces diverge
  2  usage error, parse error, or a dump/hash integrity failure

USAGE
=====
  python tools/identity_trace_diff.py A.trace B.trace [--labels APPLE,NVIDIA]
                                      [--max-cells N] [--all] [--no-verify-dumps]
  python tools/identity_trace_diff.py --selftest

Pure standard library: struct, difflib, argparse. No numpy, no pandas.
Output ordering is deterministic.
"""

import argparse
import difflib
import glob
import io
import os
import re
import struct
import sys
import tempfile
from collections import Counter

# --------------------------------------------------------------------------
# FNV-1a 64
# --------------------------------------------------------------------------

FNV64_OFFSET_BASIS = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data):
    """FNV-1a 64-bit, byte at a time, over `data` (bytes-like)."""
    h = FNV64_OFFSET_BASIS
    for b in data:
        h ^= b
        h = (h * FNV64_PRIME) & _MASK64
    return h


def fnv1a64_hex(data):
    return "%016x" % fnv1a64(data)


# --------------------------------------------------------------------------
# dtype table
# --------------------------------------------------------------------------

# name -> (struct code for the VALUE, itemsize, kind, struct code for the BITS)
DTYPES = {
    "f32": ("<f", 4, "float", "<I"),
    "f64": ("<d", 8, "float", "<Q"),
    "u32": ("<I", 4, "uint", "<I"),
    "i32": ("<i", 4, "int", "<I"),
    "u8": ("<B", 1, "uint", "<B"),
    "i64": ("<q", 8, "int", "<Q"),
}

_HASH_RE = re.compile(r"\A[0-9a-f]{16}\Z")
_INT_RE = re.compile(r"\A(?:0|[1-9][0-9]*)\Z")


class TraceError(Exception):
    """A hard, non-recoverable problem with an input file."""


class Record(object):
    __slots__ = ("seq", "tag", "dtype", "count", "hash", "lineno")

    def __init__(self, seq, tag, dtype, count, hsh, lineno):
        self.seq = seq
        self.tag = tag
        self.dtype = dtype
        self.count = count
        self.hash = hsh
        self.lineno = lineno

    def __repr__(self):
        return "Record(seq=%d, tag=%r, dtype=%s, count=%d, hash=%s)" % (
            self.seq, self.tag, self.dtype, self.count, self.hash)


def parse_trace(path):
    """Parse a trace file into a list of Record. Raises TraceError."""
    if not os.path.isfile(path):
        raise TraceError("%s: not a file (or does not exist)" % path)
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            raw = fh.read()
    except UnicodeDecodeError as exc:
        raise TraceError("%s: not valid UTF-8 (%s)" % (path, exc))
    except OSError as exc:
        raise TraceError("%s: cannot read (%s)" % (path, exc))

    records = []
    expected_seq = 0
    n_comment = 0
    n_blank = 0
    for lineno, line in enumerate(raw.split("\n"), start=1):
        line = line.rstrip("\r")
        if line == "":
            n_blank += 1
            continue
        if line.startswith("#"):
            n_comment += 1
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise TraceError(
                "%s:%d: malformed record: expected exactly 5 TAB separated "
                "fields, got %d" % (path, lineno, len(fields)))
        seq_s, tag, dtype, count_s, hsh = fields

        if not _INT_RE.match(seq_s):
            raise TraceError("%s:%d: malformed seq %r (want a decimal integer)"
                             % (path, lineno, seq_s))
        seq = int(seq_s)
        if seq != expected_seq:
            raise TraceError(
                "%s:%d: seq out of order: expected %d, got %d "
                "(seq must be 0-based and increase by exactly 1)"
                % (path, lineno, expected_seq, seq))

        if tag == "":
            raise TraceError("%s:%d: empty tag" % (path, lineno))
        try:
            tag.encode("ascii")
        except UnicodeEncodeError:
            raise TraceError("%s:%d: tag is not ASCII: %r" % (path, lineno, tag))
        if any(c in tag for c in ("\t", "\n", "\r")):
            raise TraceError("%s:%d: tag contains whitespace control chars: %r"
                             % (path, lineno, tag))

        if dtype not in DTYPES:
            raise TraceError("%s:%d: unknown dtype %r (want one of: %s)"
                             % (path, lineno, dtype,
                                " ".join(sorted(DTYPES))))

        if not _INT_RE.match(count_s):
            raise TraceError("%s:%d: malformed count %r (want a decimal integer)"
                             % (path, lineno, count_s))
        count = int(count_s)

        if not _HASH_RE.match(hsh):
            raise TraceError(
                "%s:%d: malformed hash %r (want exactly 16 lowercase hex digits)"
                % (path, lineno, hsh))

        records.append(Record(seq, tag, dtype, count, hsh, lineno))
        expected_seq += 1

    return records, n_comment, n_blank


# --------------------------------------------------------------------------
# dump lookup and verification
# --------------------------------------------------------------------------

def sanitize_tag(tag):
    """Filename-safe form of a tag. Anything outside [A-Za-z0-9._-] becomes '_'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def find_dump(trace_path, rec):
    """Locate the .bin dump for `rec`, if any.

    Returns (path, note). `path` is None when no dump was found or when the
    match was ambiguous; `note` explains what happened. This never guesses
    between multiple candidates.
    """
    cand_sanitized = "%s.%d.%s.bin" % (trace_path, rec.seq, sanitize_tag(rec.tag))
    if os.path.isfile(cand_sanitized):
        return cand_sanitized, "exact (sanitized tag)"
    cand_raw = "%s.%d.%s.bin" % (trace_path, rec.seq, rec.tag)
    if cand_raw != cand_sanitized and os.path.isfile(cand_raw):
        return cand_raw, "exact (raw tag)"
    pattern = "%s.%d.*.bin" % (trace_path, rec.seq)
    hits = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
    if len(hits) == 1:
        return hits[0], "matched by seq glob %s" % pattern
    if len(hits) > 1:
        return None, ("AMBIGUOUS: %d files match %s (%s); refusing to guess"
                      % (len(hits), pattern, ", ".join(os.path.basename(h) for h in hits)))
    return None, "no dump found (tried %s and %s)" % (
        os.path.basename(cand_sanitized), pattern)


def read_dump(path, rec):
    """Read a dump and check its length against the record. Raises TraceError."""
    itemsize = DTYPES[rec.dtype][1]
    expect = rec.count * itemsize
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise TraceError("%s: cannot read dump (%s)" % (path, exc))
    if len(data) != expect:
        raise TraceError(
            "%s: dump size %d bytes does not match record seq=%d tag=%r "
            "dtype=%s count=%d (expected %d bytes)"
            % (path, len(data), rec.seq, rec.tag, rec.dtype, rec.count, expect))
    return data


def verify_dumps(trace_path, records, label, out):
    """Re-hash every dump found beside `trace_path`. Returns (n_checked, failures).

    A failure means the writer and this reader disagree about the format, which
    voids every other conclusion, so the caller must exit 2.
    """
    n_checked = 0
    failures = []
    for rec in records:
        path, _note = find_dump(trace_path, rec)
        if path is None:
            continue
        try:
            data = read_dump(path, rec)
        except TraceError as exc:
            failures.append(str(exc))
            continue
        got = fnv1a64_hex(data)
        n_checked += 1
        if got != rec.hash:
            failures.append(
                "%s: dump hashes to %s but record seq=%d tag=%r records %s"
                % (path, got, rec.seq, rec.tag, rec.hash))
    if n_checked or failures:
        out("  %s: verified %d dump(s) against their recorded hashes, %d failure(s)"
            % (label, n_checked, len(failures)))
    return n_checked, failures


# --------------------------------------------------------------------------
# float bit surgery
# --------------------------------------------------------------------------

def _float_parts(bits, nbits):
    if nbits == 32:
        sign_shift, exp_bits, mant_bits = 31, 8, 23
    else:
        sign_shift, exp_bits, mant_bits = 63, 11, 52
    sign = (bits >> sign_shift) & 1
    exp = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mant = bits & ((1 << mant_bits) - 1)
    exp_max = (1 << exp_bits) - 1
    return sign, exp, mant, exp_max


def _is_nan(bits, nbits):
    _s, exp, mant, exp_max = _float_parts(bits, nbits)
    return exp == exp_max and mant != 0


def _is_inf(bits, nbits):
    _s, exp, mant, exp_max = _float_parts(bits, nbits)
    return exp == exp_max and mant == 0


def _is_zero(bits, nbits):
    _s, exp, mant, _m = _float_parts(bits, nbits)
    return exp == 0 and mant == 0


def _is_subnormal(bits, nbits):
    _s, exp, mant, _m = _float_parts(bits, nbits)
    return exp == 0 and mant != 0


def _ordinal(bits, nbits):
    """Monotone ordering key over IEEE754 bit patterns. +0 and -0 both map to 0."""
    sign_mask = 1 << (nbits - 1)
    if bits & sign_mask:
        return -(bits & (sign_mask - 1))
    return bits


_ULP_BUCKETS = (1, 2, 4, 8, 16, 64, 256, 4096, 65536)


def classify_float(bits_a, bits_b, nbits):
    """Return (class_name, signed_ulp_or_None).

    Signed ULP is ordinal(B) - ordinal(A); it is None where undefined (NaN or
    infinity on either side).
    """
    if bits_a == bits_b:
        return "IDENTICAL", 0
    a_nan = _is_nan(bits_a, nbits)
    b_nan = _is_nan(bits_b, nbits)
    if a_nan and b_nan:
        return "DIFFERENT-NAN-PAYLOAD", None
    if a_nan or b_nan:
        return "NAN-vs-NUMBER", None
    a_zero = _is_zero(bits_a, nbits)
    b_zero = _is_zero(bits_b, nbits)
    a_sub = _is_subnormal(bits_a, nbits)
    b_sub = _is_subnormal(bits_b, nbits)
    if (a_zero and b_sub) or (b_zero and a_sub):
        # Metal flushes subnormals to zero; CUDA does not by default.
        return "DENORMAL-vs-ZERO", _ordinal(bits_b, nbits) - _ordinal(bits_a, nbits)
    a_inf = _is_inf(bits_a, nbits)
    b_inf = _is_inf(bits_b, nbits)
    if a_inf and b_inf:
        return "SIGN", None          # both infinite, bits differ => opposite signs
    if a_inf or b_inf:
        return "LARGE", None
    sign_mask = 1 << (nbits - 1)
    ulp = _ordinal(bits_b, nbits) - _ordinal(bits_a, nbits)
    if (bits_a & sign_mask) != (bits_b & sign_mask):
        # covers +0 vs -0 as well, where the magnitudes are equal
        return "SIGN", ulp
    mag = abs(ulp)
    for n in _ULP_BUCKETS:
        if mag <= n:
            return "ULP<=%d" % n, ulp
    return "LARGE", ulp


def classify_int(bits_a, bits_b):
    if bits_a == bits_b:
        return "IDENTICAL", None
    # Integer buffers have no tolerance model. Bit-exactness is the only
    # meaningful relation, so there is exactly one non-identical class.
    return "DIFFERENT", None


def _fmt_float(v):
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return "%.17g" % v


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

class Alignment(object):
    def __init__(self):
        self.pairs = []          # [(rec_a, rec_b)] from 'equal' blocks, in order
        self.a_only = []         # [(index_in_a, tag)]
        self.b_only = []
        self.replace_blocks = [] # [(a_tags, b_tags)]
        self.inversions = []     # [(tag_x, tag_y)] x before y in A, y before x in B
        self.inversion_scan_truncated = False
        self.n_reordered_tags = 0


def align(recs_a, recs_b):
    tags_a = [r.tag for r in recs_a]
    tags_b = [r.tag for r in recs_b]
    al = Alignment()
    sm = difflib.SequenceMatcher(None, tags_a, tags_b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                al.pairs.append((recs_a[i1 + k], recs_b[j1 + k]))
        elif op == "delete":
            al.a_only.extend((i, tags_a[i]) for i in range(i1, i2))
        elif op == "insert":
            al.b_only.extend((j, tags_b[j]) for j in range(j1, j2))
        elif op == "replace":
            al.a_only.extend((i, tags_a[i]) for i in range(i1, i2))
            al.b_only.extend((j, tags_b[j]) for j in range(j1, j2))
            al.replace_blocks.append((tags_a[i1:i2], tags_b[j1:j2]))

    # Relative-order check over tags that occur exactly once on BOTH sides.
    # Those are the only tags for which "the order differs" is unambiguous.
    ca = Counter(tags_a)
    cb = Counter(tags_b)
    pos_a = {}
    pos_b = {}
    for i, t in enumerate(tags_a):
        pos_a.setdefault(t, i)
    for j, t in enumerate(tags_b):
        pos_b.setdefault(t, j)
    uniq = [t for t in tags_a if ca[t] == 1 and cb.get(t, 0) == 1]
    order = sorted(uniq, key=lambda t: pos_a[t])
    budget = 200000
    inverted_tags = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            budget -= 1
            if budget < 0:
                al.inversion_scan_truncated = True
                break
            if pos_b[order[i]] > pos_b[order[j]]:
                inverted_tags.add(order[i])
                inverted_tags.add(order[j])
                if len(al.inversions) < 20:
                    al.inversions.append((order[i], order[j]))
        if budget < 0:
            break
    al.n_reordered_tags = len(inverted_tags)
    return al


# --------------------------------------------------------------------------
# cell-level comparison
# --------------------------------------------------------------------------

def compare_cells(rec_a, rec_b, data_a, data_b, max_cells, out):
    """Per-cell comparison of two dumps for the same record. Returns n_diff."""
    vfmt, itemsize, kind, bfmt = DTYPES[rec_a.dtype]
    count = rec_a.count
    nbits = itemsize * 8
    hexw = itemsize * 2

    diffs = []           # (index, bits_a, bits_b, cls, ulp)
    class_counts = Counter()
    n_diff = 0
    for i in range(count):
        off = i * itemsize
        ba = struct.unpack_from(bfmt, data_a, off)[0]
        bb = struct.unpack_from(bfmt, data_b, off)[0]
        if ba == bb:
            continue
        n_diff += 1
        if kind == "float":
            cls, ulp = classify_float(ba, bb, nbits)
        else:
            cls, ulp = classify_int(ba, bb)
        class_counts[cls] += 1
        if len(diffs) < max_cells:
            diffs.append((i, ba, bb, cls, ulp))

    out("")
    out("  CELL-LEVEL COMPARISON")
    out("    differing elements: %d of %d (%.6f%%)"
        % (n_diff, count, (100.0 * n_diff / count) if count else 0.0))
    if n_diff == 0:
        out("    NOTE: the dumps are bit-identical even though the recorded")
        out("          hashes differ. That is a writer bug or a stale dump;")
        out("          do not trust either number until it is explained.")
        return n_diff

    out("    first %d differing cell(s):" % len(diffs))
    for (i, ba, bb, cls, ulp) in diffs:
        if kind == "float":
            va = struct.unpack(vfmt, struct.pack(bfmt, ba))[0]
            vb = struct.unpack(vfmt, struct.pack(bfmt, bb))[0]
            va_s = " (%s)" % _fmt_float(va)
            vb_s = " (%s)" % _fmt_float(vb)
        else:
            va = struct.unpack(vfmt, struct.pack(bfmt, ba))[0]
            vb = struct.unpack(vfmt, struct.pack(bfmt, bb))[0]
            va_s = " (%d)" % va
            vb_s = " (%d)" % vb
        ulp_s = ("ulp=%+d" % ulp) if ulp is not None else "ulp=n/a"
        out("      [%d] A 0x%0*x%s | B 0x%0*x%s | %s | %s"
            % (i, hexw, ba, va_s, hexw, bb, vb_s, ulp_s, cls))
    if n_diff > len(diffs):
        out("      ... %d further differing cell(s) not listed (--max-cells %d)"
            % (n_diff - len(diffs), max_cells))

    out("")
    out("    CLASS SUMMARY (differing cells only)")
    for cls, n in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        out("      %-24s %8d  (%6.2f%% of differing cells)"
            % (cls, n, 100.0 * n / n_diff))

    # The one interpretation worth stating outright.
    if class_counts.get("DENORMAL-vs-ZERO", 0) == n_diff:
        out("")
        out("    DIAGNOSIS: EVERY differing cell is DENORMAL-vs-ZERO. This is the")
        out("    flush-to-zero divergence: Metal flushes subnormals to zero,")
        out("    CUDA does not by default. This is a mode difference, not a")
        out("    numeric-accuracy problem, and it is not ULP noise.")
    elif class_counts.get("DENORMAL-vs-ZERO", 0):
        out("")
        out("    NOTE: %d cell(s) are DENORMAL-vs-ZERO (Metal flush-to-zero vs"
            % class_counts["DENORMAL-vs-ZERO"])
        out("    CUDA), mixed with other classes. Treat the flush-to-zero cells")
        out("    as a separate finding from the rest.")
    return n_diff


# --------------------------------------------------------------------------
# main comparison driver
# --------------------------------------------------------------------------

def run(argv):
    """Run the tool. Returns the process exit code."""
    p = argparse.ArgumentParser(
        prog="identity_trace_diff.py",
        description="Locate the first diverging stage between two identity traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="*", metavar="TRACE",
                   help="exactly two trace files: A.trace B.trace")
    p.add_argument("--labels", default=None,
                   help="comma separated pair of labels, e.g. APPLE,NVIDIA")
    p.add_argument("--max-cells", type=int, default=10,
                   help="max differing cells to list in detail (default 10)")
    p.add_argument("--all", action="store_true",
                   help="list every diverging record, not just the first")
    p.add_argument("--no-verify-dumps", action="store_true",
                   help="skip re-hashing every .bin dump found beside the traces")
    p.add_argument("--selftest", action="store_true",
                   help="build temporary traces and check the tool against them")

    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0

    if args.selftest:
        return _selftest()

    buf = []

    def out(s=""):
        buf.append(s)

    if len(args.traces) != 2:
        sys.stderr.write("usage error: expected exactly 2 trace files, got %d\n"
                         % len(args.traces))
        sys.stderr.write(p.format_usage())
        return 2
    if args.max_cells < 0:
        sys.stderr.write("usage error: --max-cells must be >= 0\n")
        return 2

    path_a, path_b = args.traces
    if args.labels is None:
        label_a, label_b = "A", "B"
    else:
        parts = args.labels.split(",")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            sys.stderr.write("usage error: --labels wants exactly two non-empty "
                             "comma separated names, e.g. --labels APPLE,NVIDIA\n")
            return 2
        label_a, label_b = parts[0].strip(), parts[1].strip()

    try:
        recs_a, com_a, blk_a = parse_trace(path_a)
        recs_b, com_b, blk_b = parse_trace(path_b)
    except TraceError as exc:
        sys.stderr.write("parse error: %s\n" % exc)
        return 2

    rc = 0

    out("=" * 78)
    out("IDENTITY TRACE DIFF")
    out("=" * 78)
    out("  %s = %s" % (label_a, os.path.abspath(path_a)))
    out("  %s = %s" % (label_b, os.path.abspath(path_b)))
    out("")
    out("STEP 1 -- PARSE")
    out("  %s: %d record(s), %d comment line(s), %d blank line(s)"
        % (label_a, len(recs_a), com_a, blk_a))
    out("  %s: %d record(s), %d comment line(s), %d blank line(s)"
        % (label_b, len(recs_b), com_b, blk_b))
    if not recs_a or not recs_b:
        out("  NOTE: at least one trace holds no records. Nothing can be")
        out("        concluded about the fit from an empty trace.")

    # Dump integrity comes first: if the writer and the reader disagree about
    # the byte format, every other conclusion below is void.
    if not args.no_verify_dumps:
        out("")
        out("STEP 1b -- DUMP INTEGRITY (re-hash every .bin found)")
        fails = []
        for pth, recs, lbl in ((path_a, recs_a, label_a), (path_b, recs_b, label_b)):
            _n, f = verify_dumps(pth, recs, lbl, out)
            fails.extend(f)
        if fails:
            out("")
            out("  !!! DUMP/HASH INTEGRITY FAILURE !!!")
            out("  The writer and this reader disagree. EVERY other conclusion")
            out("  in this report is void until this is resolved.")
            for f in fails:
                out("    %s" % f)
            print("\n".join(buf))
            return 2

    # ---- step 2: alignment -------------------------------------------------
    al = align(recs_a, recs_b)
    tags_a = [r.tag for r in recs_a]
    tags_b = [r.tag for r in recs_b]
    same_sequence = (tags_a == tags_b)

    out("")
    out("STEP 2 -- ALIGNMENT (tag sequence, computed BEFORE any hash compare)")
    out("  matched stage pairs: %d" % len(al.pairs))
    out("  unmatched in %s: %d    unmatched in %s: %d"
        % (label_a, len(al.a_only), label_b, len(al.b_only)))

    if same_sequence:
        out("  TAG SEQUENCES ARE IDENTICAL.")
    else:
        rc = max(rc, 1)
        out("")
        out("  " + "*" * 72)
        out("  *** TAG SEQUENCES DIFFER ***")
        out("  The two runs did not emit the same stages in the same order.")
        out("  That normally means THE TWO BACKENDS TOOK DIFFERENT CODE PATHS,")
        out("  which is a bigger finding than any hash difference below.")
        out("  A hash comparison across differing stage sets is NOT MEANINGFUL")
        out("  ON ITS OWN: matched blocks may line up stages that were reached")
        out("  under different conditions. Resolve the path difference first.")
        out("  The hash walk in STEP 3 still runs, over the matched blocks only.")
        out("  " + "*" * 72)

        def _uniq_first20(items):
            seen = set()
            res = []
            for idx, t in items:
                if t in seen:
                    continue
                seen.add(t)
                res.append((idx, t))
                if len(res) == 20:
                    break
            return res

        out("")
        out("  Tags present in %s but not matched in %s (first 20 distinct):"
            % (label_a, label_b))
        only_a = _uniq_first20(al.a_only)
        if not only_a:
            out("    (none)")
        for idx, t in only_a:
            out("    seq %-6d %s" % (recs_a[idx].seq, t))
        if len(set(t for _i, t in al.a_only)) > len(only_a):
            out("    ... %d more distinct"
                % (len(set(t for _i, t in al.a_only)) - len(only_a)))

        out("")
        out("  Tags present in %s but not matched in %s (first 20 distinct):"
            % (label_b, label_a))
        only_b = _uniq_first20(al.b_only)
        if not only_b:
            out("    (none)")
        for idx, t in only_b:
            out("    seq %-6d %s" % (recs_b[idx].seq, t))
        if len(set(t for _i, t in al.b_only)) > len(only_b):
            out("    ... %d more distinct"
                % (len(set(t for _i, t in al.b_only)) - len(only_b)))

        out("")
        if al.inversions:
            out("  ORDER DIFFERENCES among tags occurring exactly once on both")
            out("  sides: %d tag(s) involved. First %d inverted pair(s)"
                % (al.n_reordered_tags, len(al.inversions)))
            out("  (X before Y in %s, Y before X in %s):" % (label_a, label_b))
            for x, y in al.inversions:
                out("    %s  <->  %s" % (x, y))
        else:
            out("  No relative-order differences detected among tags occurring")
            out("  exactly once on both sides.")
        if al.inversion_scan_truncated:
            out("  NOTE: the order scan hit its comparison budget and stopped")
            out("        early. The list above is incomplete; treat the absence")
            out("        of further inversions as UNKNOWN, not as absence.")
        if al.replace_blocks:
            perm = [(a, b) for (a, b) in al.replace_blocks
                    if sorted(a) == sorted(b)]
            if perm:
                out("")
                out("  %d replaced block(s) hold the SAME multiset of tags in a"
                    % len(perm))
                out("  different order (a pure reordering). First 3:")
                for a, b in perm[:3]:
                    out("    %s: %s" % (label_a, ", ".join(a[:8])))
                    out("    %s: %s" % (label_b, ", ".join(b[:8])))

    # ---- step 3: hash walk over matched pairs ------------------------------
    out("")
    out("STEP 3 -- HASH WALK over %d matched pair(s)" % len(al.pairs))

    first_div = None          # (rec_a, rec_b, context_tags)
    all_divs = []
    shape_mismatch = None
    dtype_mismatch = None
    context = []
    n_equal = 0

    for rec_a, rec_b in al.pairs:
        if rec_a.dtype != rec_b.dtype:
            # Not in the original spec, but a dtype change at the same tag is a
            # structural difference exactly like a count change, and comparing
            # hashes across it would be meaningless.
            dtype_mismatch = (rec_a, rec_b, list(context[-3:]))
            break
        if rec_a.count != rec_b.count:
            shape_mismatch = (rec_a, rec_b, list(context[-3:]))
            break
        if rec_a.hash != rec_b.hash:
            if first_div is None:
                first_div = (rec_a, rec_b, list(context[-3:]))
            all_divs.append((rec_a, rec_b))
            if not args.all:
                break
            continue
        n_equal += 1
        context.append(rec_a.tag)

    if dtype_mismatch is not None:
        rc = max(rc, 1)
        ra, rb, ctx = dtype_mismatch
        out("  %d matched pair(s) agreed before this point." % n_equal)
        _print_context(out, ctx, label_a, label_b)
        out("")
        out("  DTYPE MISMATCH: %s" % ra.tag)
        out("    %s: seq=%d dtype=%s count=%d" % (label_a, ra.seq, ra.dtype, ra.count))
        out("    %s: seq=%d dtype=%s count=%d" % (label_b, rb.seq, rb.dtype, rb.count))
        out("  The same stage hashed different element types. This is a")
        out("  STRUCTURAL difference, not a numeric one. Stopping: nothing")
        out("  after this point can be compared meaningfully.")
    elif shape_mismatch is not None:
        rc = max(rc, 1)
        ra, rb, ctx = shape_mismatch
        out("  %d matched pair(s) agreed before this point." % n_equal)
        _print_context(out, ctx, label_a, label_b)
        out("")
        out("  SHAPE MISMATCH: %s" % ra.tag)
        out("    %s: seq=%d count=%d" % (label_a, ra.seq, ra.count))
        out("    %s: seq=%d count=%d" % (label_b, rb.seq, rb.count))
        out("    delta: %+d element(s)" % (rb.count - ra.count))
        out("  The same stage hashed a different number of elements. This is a")
        out("  STRUCTURAL difference, not a numeric one: the two runs built")
        out("  different amounts of work at this stage (different histogram")
        out("  width, different node count, different sampled row count).")
        out("  Stopping. Chasing bit patterns before this is resolved wastes")
        out("  the investigation.")
    elif first_div is not None:
        rc = max(rc, 1)
        ra, rb, ctx = first_div
        out("  %d matched pair(s) agreed before the first divergence." % n_equal)
        _print_context(out, ctx, label_a, label_b)
        out("")
        out("  FIRST DIVERGENCE: %s" % ra.tag)
        lw = max(len(label_a), len(label_b))
        out("    %-*s seq=%-6d dtype=%-3s count=%-10d hash=%s"
            % (lw + 1, label_a + ":", ra.seq, ra.dtype, ra.count, ra.hash))
        out("    %-*s seq=%-6d dtype=%-3s count=%-10d hash=%s"
            % (lw + 1, label_b + ":", rb.seq, rb.dtype, rb.count, rb.hash))
        if args.all:
            out("")
            out("  ALL DIVERGING RECORDS (%d):" % len(all_divs))
            for da, db in all_divs:
                out("    %-50s %s(seq %d) %s != %s(seq %d) %s"
                    % (da.tag, label_a, da.seq, da.hash, label_b, db.seq, db.hash))
        # ---- step 4: cell level -------------------------------------------
        rc = max(rc, _cell_stage(path_a, path_b, ra, rb, label_a, label_b,
                                 args.max_cells, out))
    else:
        if al.pairs:
            out("  All %d matched pair(s) agree on dtype, count and hash." % len(al.pairs))
        else:
            out("  No matched pairs at all. The two traces share no common stage")
            out("  subsequence; there is nothing to compare.")
            rc = max(rc, 1)

    out("")
    out("=" * 78)
    if rc == 0:
        out("RESULT: IDENTICAL. Same stage sequence, same counts, same hashes.")
        out("  Caveat: a matching hash means the buffers agreed at each")
        out("  checkpoint, NOT that the computation was identical, and nothing")
        out("  is known about whatever the trace did not checkpoint.")
    else:
        out("RESULT: DIVERGENT.")
    out("=" * 78)

    print("\n".join(buf))
    return rc


def _print_context(out, ctx, label_a, label_b):
    out("")
    out("  Context, the last %d stage(s) that were still IDENTICAL on both sides:"
        % len(ctx))
    if not ctx:
        out("    (none: the very first matched stage already differs)")
    for t in ctx:
        out("    OK  %s" % t)


def _cell_stage(path_a, path_b, ra, rb, label_a, label_b, max_cells, out):
    """Load the .bin dumps for the first diverging record and compare cells.

    Returns an exit-code contribution (2 on an integrity failure, else 0).
    """
    out("")
    out("STEP 4 -- CELL LEVEL (requires a .bin dump on BOTH sides)")
    dump_a, note_a = find_dump(path_a, ra)
    dump_b, note_b = find_dump(path_b, rb)
    out("  %s dump: %s" % (label_a, dump_a if dump_a else "NOT USABLE -- " + note_a))
    out("  %s dump: %s" % (label_b, dump_b if dump_b else "NOT USABLE -- " + note_b))
    if dump_a is None or dump_b is None:
        out("  Cell-level comparison SKIPPED: a dump is missing or ambiguous on")
        out("  at least one side. Which cells differ, and whether the difference")
        out("  is flush-to-zero or ULP noise, is UNKNOWN from this run.")
        return 0
    try:
        data_a = read_dump(dump_a, ra)
        data_b = read_dump(dump_b, rb)
    except TraceError as exc:
        out("  DUMP ERROR: %s" % exc)
        out("  Cell-level comparison SKIPPED.")
        return 0
    got_a = fnv1a64_hex(data_a)
    got_b = fnv1a64_hex(data_b)
    bad = []
    if got_a != ra.hash:
        bad.append("%s dump hashes to %s, record says %s" % (label_a, got_a, ra.hash))
    if got_b != rb.hash:
        bad.append("%s dump hashes to %s, record says %s" % (label_b, got_b, rb.hash))
    if bad:
        out("  !!! DUMP/HASH INTEGRITY FAILURE on the diverging record !!!")
        for b in bad:
            out("    %s" % b)
        out("  The writer and this reader disagree about the byte format.")
        out("  EVERY conclusion above is void.")
        return 2
    out("  Both dumps re-hash to their recorded hashes. Bytes trusted.")
    compare_cells(ra, rb, data_a, data_b, max_cells, out)
    return 0


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------

def _mk_trace(path, records):
    """records: [(tag, dtype, count, hash_hex)] -- seq assigned in order."""
    lines = ["# identity trace (selftest fixture)"]
    for i, (tag, dtype, count, hsh) in enumerate(records):
        lines.append("%d\t%s\t%s\t%d\t%s" % (i, tag, dtype, count, hsh))
        if i == 0:
            lines.append("")            # blank line, must be skipped
            lines.append("# midstream comment, must be skipped")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _pack_f32(bit_list):
    return b"".join(struct.pack("<I", b) for b in bit_list)


def _f32bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _capture(argv):
    """Run(argv) with stdout captured. Returns (rc, text)."""
    sio = io.StringIO()
    old = sys.stdout
    sys.stdout = sio
    try:
        rc = run(argv)
    finally:
        sys.stdout = old
    return rc, sio.getvalue()


def _selftest():
    tmp = tempfile.mkdtemp(prefix="identity_trace_diff_selftest_")
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print("%-46s %s%s" % (name, "PASS" if cond else "FAIL",
                              ("  <- " + detail) if (detail and not cond) else ""))

    print("identity_trace_diff selftest")
    print("workdir: %s" % tmp)
    print("-" * 78)

    # ---- case 0: FNV-1a64 known answers --------------------------------
    kats = [(b"", "cbf29ce484222325"),
            (b"a", "af63dc4c8601ec8c"),
            (b"foobar", "85944171f73967e8")]
    ok = all(fnv1a64_hex(d) == e for d, e in kats)
    check("fnv1a64 known answers", ok,
          "got " + ", ".join(fnv1a64_hex(d) for d, _e in kats))

    base = [("tree00.init.gpair", "f32", 4, "0000000000000001"),
            ("tree00.level00.hist.after_scan", "f32", 8, "0000000000000002"),
            ("tree00.level00.hist.after_reduce", "f32", 8, "0000000000000003"),
            ("tree00.level00.split.best", "u32", 2, "0000000000000004"),
            ("tree00.level01.hist.after_scan", "f32", 16, "0000000000000005"),
            ("tree00.leaf.values", "f32", 4, "0000000000000006")]

    # ---- case 1: identical ---------------------------------------------
    a1 = os.path.join(tmp, "c1a.trace")
    b1 = os.path.join(tmp, "c1b.trace")
    _mk_trace(a1, base)
    _mk_trace(b1, base)
    rc, txt = _capture([a1, b1, "--labels", "APPLE,NVIDIA"])
    check("case1 identical -> exit 0", rc == 0, "rc=%d" % rc)
    check("case1 says IDENTICAL", "RESULT: IDENTICAL" in txt)
    check("case1 tag sequences identical",
          "TAG SEQUENCES ARE IDENTICAL." in txt)

    # ---- case 2: one hash divergence in the middle ----------------------
    mid = list(base)
    mid[3] = (mid[3][0], mid[3][1], mid[3][2], "00000000000000ff")
    a2 = os.path.join(tmp, "c2a.trace")
    b2 = os.path.join(tmp, "c2b.trace")
    _mk_trace(a2, base)
    _mk_trace(b2, mid)
    rc, txt = _capture([a2, b2])
    check("case2 mid divergence -> exit 1", rc == 1, "rc=%d" % rc)
    check("case2 first diverging tag",
          "FIRST DIVERGENCE: tree00.level00.split.best" in txt)
    check("case2 shows 3 lines of matching context",
          txt.count("    OK  ") == 3, "count=%d" % txt.count("    OK  "))
    check("case2 context is the 3 preceding tags",
          "OK  tree00.level00.hist.after_reduce" in txt
          and "OK  tree00.init.gpair" in txt)

    # ---- case 3: shape mismatch ----------------------------------------
    shp = list(base)
    shp[2] = (shp[2][0], shp[2][1], 9, shp[2][3])
    a3 = os.path.join(tmp, "c3a.trace")
    b3 = os.path.join(tmp, "c3b.trace")
    _mk_trace(a3, base)
    _mk_trace(b3, shp)
    rc, txt = _capture([a3, b3])
    check("case3 shape mismatch -> exit 1", rc == 1, "rc=%d" % rc)
    check("case3 reports SHAPE MISMATCH tag",
          "SHAPE MISMATCH: tree00.level00.hist.after_reduce" in txt)
    check("case3 does not claim a hash divergence",
          "FIRST DIVERGENCE:" not in txt)
    check("case3 stops at the shape mismatch",
          "Stopping." in txt)

    # ---- case 4: a tag present on one side only -------------------------
    extra = list(base)
    extra.insert(3, ("tree00.level00.hist.apple_only_fixup", "f32", 8,
                     "00000000000000aa"))
    a4 = os.path.join(tmp, "c4a.trace")
    b4 = os.path.join(tmp, "c4b.trace")
    _mk_trace(a4, extra)
    _mk_trace(b4, base)
    rc, txt = _capture([a4, b4, "--labels", "APPLE,NVIDIA"])
    check("case4 extra tag -> exit 1", rc == 1, "rc=%d" % rc)
    check("case4 says TAG SEQUENCES DIFFER", "*** TAG SEQUENCES DIFFER ***" in txt)
    check("case4 lists the A-only tag",
          "tree00.level00.hist.apple_only_fixup" in txt)
    check("case4 warns hashes are not meaningful alone",
          "NOT MEANINGFUL" in txt)
    check("case4 still walks matched blocks with no hash divergence",
          "FIRST DIVERGENCE:" not in txt and "SHAPE MISMATCH:" not in txt)

    # ---- case 5: a reordered tag ----------------------------------------
    ro = list(base)
    ro[1], ro[2] = ro[2], ro[1]
    a5 = os.path.join(tmp, "c5a.trace")
    b5 = os.path.join(tmp, "c5b.trace")
    _mk_trace(a5, base)
    _mk_trace(b5, ro)
    rc, txt = _capture([a5, b5])
    check("case5 reorder -> exit 1", rc == 1, "rc=%d" % rc)
    check("case5 says TAG SEQUENCES DIFFER", "*** TAG SEQUENCES DIFFER ***" in txt)
    check("case5 reports the inverted pair",
          "tree00.level00.hist.after_scan  <->  tree00.level00.hist.after_reduce"
          in txt)

    # ---- case 6: cell level, denormal-vs-zero + NaN payload -------------
    # 8 f32 cells:
    #  0 identical, 1 one ULP, 2 denormal vs zero, 3 different NaN payload,
    #  4 NaN vs number, 5 sign flip, 6 large, 7 identical
    cells_a = [_f32bits(1.0), _f32bits(1.0), 0x00000001, 0x7FC00000,
               0x7FC00000, _f32bits(1.0), _f32bits(1.0), _f32bits(2.5)]
    cells_b = [_f32bits(1.0), _f32bits(1.0) + 1, 0x00000000, 0x7FC00001,
               _f32bits(1.0), _f32bits(-1.0), _f32bits(1e30), _f32bits(2.5)]
    data_a = _pack_f32(cells_a)
    data_b = _pack_f32(cells_b)
    ha = fnv1a64_hex(data_a)
    hb = fnv1a64_hex(data_b)
    cell_tag = "tree00.level00.hist.after_scan"
    ca_recs = list(base)
    cb_recs = list(base)
    ca_recs[1] = (cell_tag, "f32", 8, ha)
    cb_recs[1] = (cell_tag, "f32", 8, hb)
    a6 = os.path.join(tmp, "c6a.trace")
    b6 = os.path.join(tmp, "c6b.trace")
    _mk_trace(a6, ca_recs)
    _mk_trace(b6, cb_recs)
    with open("%s.1.%s.bin" % (a6, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(data_a)
    with open("%s.1.%s.bin" % (b6, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(data_b)
    rc, txt = _capture([a6, b6, "--max-cells", "10", "--labels", "APPLE,NVIDIA"])
    check("case6 cell level -> exit 1", rc == 1, "rc=%d" % rc)
    check("case6 first diverging tag",
          "FIRST DIVERGENCE: %s" % cell_tag in txt)
    check("case6 counts 6 of 8 differing cells",
          "differing elements: 6 of 8" in txt)
    check("case6 flags DENORMAL-vs-ZERO", "DENORMAL-vs-ZERO" in txt)
    check("case6 flags DIFFERENT-NAN-PAYLOAD", "DIFFERENT-NAN-PAYLOAD" in txt)
    check("case6 flags NAN-vs-NUMBER", "NAN-vs-NUMBER" in txt)
    check("case6 flags SIGN", "| SIGN" in txt)
    check("case6 flags LARGE", "| LARGE" in txt)
    check("case6 flags ULP<=1", "ULP<=1" in txt)
    check("case6 prints the class summary",
          "CLASS SUMMARY (differing cells only)" in txt)
    check("case6 verified dumps against recorded hashes",
          "Both dumps re-hash to their recorded hashes." in txt)

    # ---- case 6b: an all-denormal divergence gets the named diagnosis ----
    cells_a2 = [0x00000001, 0x00000002, 0x00000003, 0x00000004]
    cells_b2 = [0x00000000, 0x00000000, 0x00000000, 0x00000000]
    da2, db2 = _pack_f32(cells_a2), _pack_f32(cells_b2)
    ca2 = list(base)
    cb2 = list(base)
    ca2[1] = (cell_tag, "f32", 4, fnv1a64_hex(da2))
    cb2[1] = (cell_tag, "f32", 4, fnv1a64_hex(db2))
    a6b = os.path.join(tmp, "c6ba.trace")
    b6b = os.path.join(tmp, "c6bb.trace")
    _mk_trace(a6b, ca2)
    _mk_trace(b6b, cb2)
    with open("%s.1.%s.bin" % (a6b, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(da2)
    with open("%s.1.%s.bin" % (b6b, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(db2)
    rc, txt = _capture([a6b, b6b])
    check("case6b all-denormal -> exit 1", rc == 1, "rc=%d" % rc)
    check("case6b names the flush-to-zero diagnosis",
          "EVERY differing cell is DENORMAL-vs-ZERO" in txt)

    # ---- case 7: dump does not hash to its recorded hash -> exit 2 ------
    a7 = os.path.join(tmp, "c7a.trace")
    b7 = os.path.join(tmp, "c7b.trace")
    _mk_trace(a7, ca_recs)
    _mk_trace(b7, cb_recs)
    with open("%s.1.%s.bin" % (a7, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(data_a)
    with open("%s.1.%s.bin" % (b7, sanitize_tag(cell_tag)), "wb") as fh:
        fh.write(data_a)          # wrong bytes for B's recorded hash
    rc, txt = _capture([a7, b7])
    check("case7 dump/hash mismatch -> exit 2", rc == 2, "rc=%d" % rc)
    check("case7 says integrity failure",
          "DUMP/HASH INTEGRITY FAILURE" in txt)
    check("case7 says other conclusions are void", "void" in txt)

    # ---- case 8: malformed input -> exit 2 ------------------------------
    a8 = os.path.join(tmp, "c8a.trace")
    with open(a8, "w", encoding="utf-8") as fh:
        fh.write("0\ttag.one\tf32\t4\t0000000000000001\n")
        fh.write("1\ttag.two\tf32\t4\tNOTAHASH\n")
    rc, _txt = _capture([a8, b1])
    check("case8 bad hash field -> exit 2", rc == 2, "rc=%d" % rc)

    a9 = os.path.join(tmp, "c9a.trace")
    with open(a9, "w", encoding="utf-8") as fh:
        fh.write("0\ttag.one\tf32\t4\t0000000000000001\n")
        fh.write("2\ttag.two\tf32\t4\t0000000000000002\n")
    rc, _txt = _capture([a9, b1])
    check("case9 non-consecutive seq -> exit 2", rc == 2, "rc=%d" % rc)

    a10 = os.path.join(tmp, "c10a.trace")
    with open(a10, "w", encoding="utf-8") as fh:
        fh.write("0\ttag.one\tf16\t4\t0000000000000001\n")
    rc, _txt = _capture([a10, b1])
    check("case10 unknown dtype -> exit 2", rc == 2, "rc=%d" % rc)

    rc, _txt = _capture([a1])
    check("case11 wrong argument count -> exit 2", rc == 2, "rc=%d" % rc)

    # ---- case 12: --all lists every diverging record --------------------
    two = list(base)
    two[1] = (two[1][0], two[1][1], two[1][2], "00000000000000b1")
    two[4] = (two[4][0], two[4][1], two[4][2], "00000000000000b2")
    a12 = os.path.join(tmp, "c12a.trace")
    b12 = os.path.join(tmp, "c12b.trace")
    _mk_trace(a12, base)
    _mk_trace(b12, two)
    rc, txt = _capture([a12, b12, "--all"])
    check("case12 --all -> exit 1", rc == 1, "rc=%d" % rc)
    check("case12 --all lists both diverging records",
          "ALL DIVERGING RECORDS (2)" in txt)
    rc, txt = _capture([a12, b12])
    check("case12 default stops at the first",
          "ALL DIVERGING RECORDS" not in txt
          and "FIRST DIVERGENCE: tree00.level00.hist.after_scan" in txt)

    print("-" * 78)
    n_fail = sum(1 for _n, ok_, _d in results if not ok_)
    print("%d check(s), %d passed, %d FAILED" %
          (len(results), len(results) - n_fail, n_fail))
    if n_fail:
        print("FAILED checks:")
        for n, ok_, d in results:
            if not ok_:
                print("  %s %s" % (n, d))
    print("fixtures left in %s" % tmp)
    return 1 if n_fail else 0


def main():
    try:
        return run(sys.argv[1:])
    except TraceError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
