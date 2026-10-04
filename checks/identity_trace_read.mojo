# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Reading an identity trace back in Mojo, for checks that raise on a
divergence (moved from `core/identity_trace.mojo` by lane cpu3-core: file
text processing belongs with the checks, not in a GPU-route module)."""


# =====================================================================
# READING A TRACE BACK, IN MOJO
#
# Added 2026-08-22 by the DEPTHWISE lane, at the foot, behind its own
# header. `tools/identity_trace_diff.py` is the real reader -- it aligns on
# tag SEQUENCES and classifies each differing cell, which is the diagnosis
# where a tag is only the location -- and nothing here competes with it.
#
# What these two functions are for is the case Python cannot serve: a MOJO
# CHECK that produces two traces in one process and has to RAISE on the
# difference. `checks/identity_trace_check.mojo` already needed exactly
# this and wrote the loop inline (`:71` and `:262-283`); the depthwise
# probe needed it second. Two inline copies is how a third gets written, so
# it lives here where both callers can reach it. THE INLINE COPY IN THAT
# CHECK CAN NOW BE DELETED -- that is the lossguide lane's call and its
# file, so it is named rather than done.
# =====================================================================


def read_trace_lines(path: String) raises -> List[String]:
    """Every RECORD line of a trace, comments and blanks dropped.

    `header()` writes `#` comments and the differ skips them, so a reader
    that keeps them compares provenance -- the dataset, the backend, the
    parameters -- and reports a divergence for two runs that agree on every
    number. Provenance is what those lines are FOR; it is not a record.
    """
    var out = List[String]()
    with open(path, "r") as fh:
        var body = fh.read()
        var cur = String("")
        for i in range(body.byte_length()):
            var c = String(body[byte=i])
            if c == "\n":
                _keep_record(cur, out)
                cur = String("")
            else:
                cur += c
        _keep_record(cur, out)
    return out^


def _keep_record(line: String, mut out: List[String]):
    """Append `line` unless it is blank or a `#` comment."""
    if line.byte_length() == 0:
        return
    if line.startswith("#"):
        return
    out.append(line)


def first_divergence(path_a: String, path_b: String) raises -> String:
    """The first record on which two traces disagree, or "" if they agree.

    Returns the two lines joined by `   VS   `, which is enough to read the
    tag, the dtype, the count and both hashes at a glance.

    A LENGTH MISMATCH IS REPORTED AS SUCH rather than compared line by line.
    Different lengths mean the two runs took a different number of stages --
    a different level count, a different number of leaves split -- which is
    a STRUCTURAL divergence, and a tag-by-tag diff past that point pairs
    records that were never meant to correspond. That is the failure mode
    `tools/identity_trace_diff.py` exists to handle properly with sequence
    alignment; this function's job is to say "go run that one".
    """
    var a = read_trace_lines(path_a)
    var b = read_trace_lines(path_b)
    if len(a) != len(b):
        return (
            String("<record counts differ: ")
            + String(len(a))
            + " vs "
            + String(len(b))
            + " -- STRUCTURAL divergence, the two runs did not take the"
            " same stages. Run tools/identity_trace_diff.py, which aligns"
            " on tag sequences instead of by position.>"
        )
    for i in range(len(a)):
        if a[i] != b[i]:
            return String(a[i]) + "   VS   " + String(b[i])
    return String("")
