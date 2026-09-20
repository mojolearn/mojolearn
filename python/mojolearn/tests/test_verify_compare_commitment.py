# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`verify --commitment` and `verify --compare --commitment-a/--commitment-b`:
the one hole the structural defenses in `--compare` cannot reach.

    cd python && python3 -m mojolearn.tests.test_verify_compare_commitment

Everything in `compare_documents` is about documents that are malformed or
that contradict themselves. None of it is about a WELL FORMED document whose
numbers were never computed by the machine it names, because the party wrote
them down after reading the other party's file. That attack is built here
first, run against the comparer with no commitments, and WATCHED TO PASS
CLEANLY AS `AGREE` -- because a check whose failure was never observed is not
a check -- and only then run again with commitments and watched to be caught.

Nothing here needs a GPU, a binding, a network or the repo.
"""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mojolearn import _verify_all as va

PKG = Path(va.__file__).resolve().parent


# ---------------------------------------------------------------- fixtures

_CELLS = [("ols", "base", "train", "3d1d7c30b12d9872"),
          ("ols", "base", "infer", "2546a13c03838433"),
          ("kmeans", "base", "train", "9f21ab0c4e77d510")]


def _doc(vendor, device, device_class, cells, commit="a809d92f2", verdict="VERIFIED"):
    """The shape `verify --all --json-out` writes, trimmed to what the
    comparer reads. Kept separate from `test_verify_all._doc` on purpose: a
    commitment covers fields that test does not carry, and importing it would
    make this file's coverage silently follow that one's."""
    rows = [dict(lane=c[0], fixture=c[1], part=c[2], value=c[3],
                 state=(c[4] if len(c) > 4 else "IDENTICAL")) for c in cells]
    return dict(
        format=va.COMPARE_INPUT_FORMAT, verdict=verdict,
        detail="verified %d of %d cell parts" % (len(rows), len(rows)),
        elapsed_s=12.5,
        lane_seconds={r["lane"]: 1.25 for r in rows},
        device=dict(mojolearn_version="0.8.6", commit=commit, commit_source="witness",
                    numeric_mode="identical", vendor=vendor, device_class=device_class,
                    device=device, cpu_model="cpu-x", requested_parallel_devices=[],
                    platform="p", python="3.14.6", numpy="2.3.1"),
        bindings=[dict(module="mojolearn._mojolearn", sha256="d" * 64, size=1)],
        verification_contract=dict(
            harness_sha256="e" * 64,
            fixtures={r["fixture"]: {"X": "input"} for r in rows},
            heldout={r["fixture"]: {"X": "held"} for r in rows},
            protocols={r["part"]: {"version": 1} for r in rows}),
        cells=rows)


def _honest_pair():
    """Two honest documents from genuinely different hardware, same bits."""
    return (_doc("metal", "Apple M2", "apple", _CELLS, commit="aaa"),
            _doc("cuda", "RTX 4090", "nvidia", _CELLS, commit="bbb"))


def _forged_from(victim, vendor="cuda", device="RTX 4090", cls="nvidia", commit="bbb"):
    """THE ATTACK, in one function so nobody has to imagine it.

    The forger never runs anything. They wait for the other party's document,
    lift its cell values -- and its verification contract, which is the only
    other thing hash agreement is read under -- and publish them under their
    own hardware. The result is a perfectly well formed evidence document that
    every structural check in `compare_documents` is happy with."""
    out = copy.deepcopy(victim)
    out.pop(va.REVEAL_KEY, None)
    out["device"].update(vendor=vendor, device=device, device_class=cls, commit=commit)
    out["elapsed_s"] = 11.0            # a forger would not copy the stopwatch
    return out


# ---------------------------------------------------- the hole, demonstrated

def test_the_attack_this_lane_exists_for_passes_without_commitments():
    """WATCH IT FAIL FIRST. A forged document that copies the other party's
    cells under its own provenance reads a clean `AGREE`, exit 0, across two
    "independent" vendors, with every existing defense satisfied: nothing is
    malformed, nothing is a duplicate row, the files are not byte-identical,
    the contracts match, no cell is MOVED and none is one-sided.

    This test asserts the HOLE, not a fix. If it ever starts failing because
    `--compare` learned to reject the forgery some other way, that is a real
    finding and this test should be rewritten to say so -- but it must never
    be deleted on the assumption that the commitment path covers it, because
    the commitment path only fires when commitments were actually exchanged.
    """
    honest, _ = _honest_pair()
    forged = _forged_from(honest)
    r = va.compare_documents(honest, forged, "honest.json", "forged.json")
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED, r["verdict"]
    assert r["agree"] == 3 and r["differ"] == 0
    assert r["provenance"]["independent"] is True, (
        "the forgery even reads as two independent hardware classes, which is the strongest "
        "thing this command can say about a pair of documents")
    assert not r["problems"] and not r["context_problems"]


def test_the_same_attack_is_caught_when_the_forger_committed_first():
    """THE FIX. The forger ran `--commitment` on their own honest document
    before the exchange -- which is the whole protocol -- and published the
    line. What they hand over afterwards is the forgery, and it does not hash
    to what they published."""
    honest, forger_own = _honest_pair()
    published = va.seal_document(forger_own)          # committed BEFORE the exchange
    honest_line = va.seal_document(honest)
    forged = _forged_from(honest)                     # ...then they copied

    r = va.compare_documents(honest, forged, "honest.json", "forged.json",
                             commitment_a=honest_line, commitment_b=published)
    assert r["verdict"] == "COMMITMENT BROKEN" and r["exit"] == va.EXIT_MISMATCH
    assert r["commitment"]["a"]["state"] == "verified"
    assert r["commitment"]["b"]["state"] in ("MISMATCH", "UNSEALED"), r["commitment"]["b"]
    # PRINT THE MATCHES, NOT THE COUNT: the reader must see which document and
    # which two values, or the verdict is a bare assertion.
    text = va.format_compare(r)
    assert "RESULT: COMMITMENT BROKEN" in text, text
    assert "forged.json" in text and published in text, text
    assert "RESULT: AGREE" not in text
    # AND THE STRONGEST SENTENCE THIS COMMAND HAS MUST NOT APPEAR OVER IT. The
    # forgery reads as two independent hardware classes precisely because the
    # provenance block is the lie; printing "two independent machines reaching
    # the same bits" above a broken commitment hands the forger the line a
    # skimmer would quote.
    assert "two independent machines reaching the" not in text, text
    assert "PROVENANCE ABOVE CANNOT BE TAKEN AT FACE VALUE" in text, text


def test_the_forger_cannot_reseal_after_copying():
    """The obvious next move: copy the cells, then run `--commitment` again to
    seal the forgery. It produces a valid document -- and a DIFFERENT
    commitment from the one already published, which is the whole point of
    publishing it first."""
    honest, forger_own = _honest_pair()
    published = va.seal_document(forger_own)
    forged = _forged_from(honest)
    resealed = va.seal_document(forged)
    assert resealed != published
    r = va.compare_documents(honest, forged, "honest.json", "forged.json",
                             commitment_b=published)
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert r["commitment"]["b"]["state"] == "MISMATCH"
    assert r["commitment"]["b"]["recomputed"] == resealed


def test_the_forger_who_keeps_their_own_nonce_and_swaps_only_the_hardware():
    """THE SHARPEST VERSION, and the one that decides what the commitment
    covers. The forger ran honestly, sealed, and published. Then, after seeing
    that the other party is on hardware whose agreement would be worth more,
    they edit the ONE thing that makes the pair look independent -- the device
    block -- and keep their own nonce and their own cells.

    A commitment over cells alone matches this exactly, and the comparison
    would read AGREE across two vendors that never both ran. It is caught only
    because the commitment covers the provenance, which is why it is held at
    the comparison level and not only at `commitment_preimage`: it is the test
    that fails if a later lane narrows what is covered.
    """
    honest, forger_own = _honest_pair()
    honest_line = va.seal_document(honest)
    published = va.seal_document(forger_own)
    assert json.dumps(forger_own["cells"], sort_keys=True) \
        == json.dumps(honest["cells"], sort_keys=True), "their run honestly agreed"

    forger_own["device"].update(vendor="rocm", device="MI300X", device_class="amd")
    r = va.compare_documents(honest, forger_own, "honest.json", "theirs.json",
                             commitment_a=honest_line, commitment_b=published)
    assert r["verdict"] == "COMMITMENT BROKEN" and r["exit"] == va.EXIT_MISMATCH, r["verdict"]
    assert r["commitment"]["b"]["state"] == "SELF-INCONSISTENT", r["commitment"]["b"]
    # and if they reseal to repair that, the published line no longer matches
    resealed = va.seal_document(forger_own)
    assert resealed != published
    r2 = va.compare_documents(honest, forger_own, "honest.json", "theirs.json",
                              commitment_a=honest_line, commitment_b=published)
    assert r2["verdict"] == "COMMITMENT BROKEN"
    assert r2["commitment"]["b"]["state"] == "MISMATCH", r2["commitment"]["b"]


def test_the_forger_who_keeps_their_own_nonce_and_swaps_only_the_contract():
    """The same move against the other half of what makes a hash comparable.
    `comparison_context_problems` refuses two documents whose fixture,
    held-out and protocol fingerprints differ, so a party who could edit the
    contract after the exchange could turn an INCOMPARABLE into an AGREE."""
    honest, forger_own = _honest_pair()
    honest_line = va.seal_document(honest)
    published = va.seal_document(forger_own)
    forger_own["verification_contract"]["fixtures"]["base"] = {"X": "theirs, copied over"}
    forger_own["verification_contract"]["harness_sha256"] = "f" * 64
    va.seal_document(forger_own)                        # repair the internal one
    r = va.compare_documents(honest, forger_own, "honest.json", "theirs.json",
                             commitment_a=honest_line, commitment_b=published)
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert r["commitment"]["b"]["state"] == "MISMATCH"


def test_a_commitment_to_cells_alone_would_not_have_caught_the_provenance_swap():
    """WHY THE COMMITMENT COVERS PROVENANCE, held as a test rather than as a
    sentence. The forgery above changes the cells' owner, not the cells. A
    commitment over `cells` only would match it exactly."""
    honest, _ = _honest_pair()
    forged = _forged_from(honest)
    cells_only = lambda d: json.dumps(d["cells"], sort_keys=True)
    assert cells_only(honest) == cells_only(forged), (
        "the forgery copies the cells verbatim; that is what makes it a forgery")
    assert va.commitment_preimage(honest) != va.commitment_preimage(forged), (
        "covering the cells alone would commit a forger to precisely the half they never "
        "needed to change")


# -------------------------------------------- what the commitment does cover

def _mutate(doc, path, value):
    out = copy.deepcopy(doc)
    node = out
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return out


_COVERED_MUTATIONS = [
    (("format",), "something-else"),
    (("cells", 0, "value"), "ffff0000ffff0000"),
    (("cells", 0, "state"), "DIVERGENT"),
    (("cells", 0, "lane"), "renamed"),
    (("cells", 0, "fixture"), "renamed"),
    (("cells", 0, "part"), "renamed"),
    (("device", "vendor"), "metal"),
    (("device", "device_class"), "apple"),
    (("device", "device"), "Apple M2"),
    (("device", "cpu_model"), "other"),
    (("device", "commit"), "deadbeef"),
    (("device", "mojolearn_version"), "9.9.9"),
    (("device", "platform"), "elsewhere"),
    (("device", "python"), "3.0.0"),
    (("device", "numeric_mode"), "fast"),
    (("verification_contract", "harness_sha256"), "f" * 64),
    (("verification_contract", "fixtures", "base"), {"X": "other"}),
    (("verification_contract", "heldout", "base"), {"X": "other"}),
    (("verification_contract", "protocols", "train"), {"version": 2}),
    (("bindings", 0, "sha256"), "c" * 64),
    (("verdict",), "MISMATCH"),
    (("detail",), "verified 1 of 999 cell parts"),
]

_EXCLUDED_MUTATIONS = [
    (("elapsed_s",), 999.0),
    (("lane_seconds", "ols"), 999.0),
]


def test_every_field_the_comparer_reads_moves_the_commitment():
    """THE RULE, run rather than written down: cover exactly the fields
    `compare_documents` reads.

    A field the comparer reads that the commitment omits is a field a party
    can still change after seeing the other document -- the hole, reopened one
    key at a time. Every mutation here is of a value `compare_documents` or
    `comparison_context_problems` actually looks at, and each must move the
    digest. PRINT THE MATCHES: a failure names the exact path that went
    uncovered.
    """
    base = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    nonce = "ab" * 16
    ref = va.commitment_digest(base, nonce)
    uncovered = [path for path, value in _COVERED_MUTATIONS
                 if va.commitment_digest(_mutate(base, path, value), nonce) == ref]
    assert not uncovered, (
        "these fields are read by the comparer and NOT covered by the commitment, so a party "
        "can change them after seeing the other document: "
        + "; ".join("/".join(map(str, p)) for p in uncovered))


def test_fields_nobody_compares_do_not_move_the_commitment():
    """The mirror rule. A commitment that fires on a field no comparison reads
    is a false alarm, and a false alarm is how a check gets normalized into
    something people click past. Wall-clock timings are the ones that would
    move on an honest rerun."""
    base = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    nonce = "ab" * 16
    ref = va.commitment_digest(base, nonce)
    for path, value in _EXCLUDED_MUTATIONS:
        assert va.commitment_digest(_mutate(base, path, value), nonce) == ref, (
            "/".join(map(str, path)) + " moved the commitment, but no comparison reads it")
    assert set(va.COMMITMENT_EXCLUDES) >= {"elapsed_s", "lane_seconds", va.REVEAL_KEY}


def test_the_covered_list_is_pinned_so_a_change_has_to_be_deliberate():
    """`COMMITMENT_COVERS` is the answer to "what does this prove". Growing or
    shrinking it silently changes that answer, so it is pinned here: a lane
    that edits it edits this line too and has to say why in the same diff."""
    assert va.COMMITMENT_COVERS == ("format", "cells", "device", "verification_contract",
                                    "bindings", "verdict", "detail")


def test_innocent_reserialization_does_not_break_a_commitment():
    """A CHECK THAT FIRES ON HONEST HANDLING IS A CHECK PEOPLE IGNORE. The
    commitment is taken over parsed values re-serialized canonically, not over
    the file's bytes, so indentation, key order and the JSON writer are all
    invisible."""
    doc = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    line = va.seal_document(doc)
    for dumped in (json.dumps(doc), json.dumps(doc, indent=4, sort_keys=True),
                   json.dumps(doc, indent=1, sort_keys=False, separators=(", ", ": "))):
        back = json.loads(dumped)
        assert va.commitment_digest(back, back[va.REVEAL_KEY]["nonce"]) == line
    # reordering the cell rows is not a change either: the comparer keys them
    # by (lane, fixture, part) and never reads their order
    shuffled = copy.deepcopy(doc)
    shuffled["cells"].reverse()
    assert va.commitment_digest(shuffled, shuffled[va.REVEAL_KEY]["nonce"]) == line


def test_a_duplicated_cell_row_is_still_covered():
    """`_read_cells` refuses a document that names one cell part twice. The
    commitment must bind a party to the document that refusal describes, not
    to a tidied one, or the malformed document and the committed document are
    two different things."""
    doc = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    nonce = "ab" * 16
    dup = copy.deepcopy(doc)
    dup["cells"].append(dict(dup["cells"][0]))
    assert va.commitment_digest(dup, nonce) != va.commitment_digest(doc, nonce)


# ----------------------------------------------------- the weaker labelling

_NO_COMMITMENT_WORDS = "COMMITMENTS: none were exchanged."


def test_a_comparison_without_commitments_still_agrees_and_says_it_is_weaker():
    """NOT A FAILURE. Two strangers with two files and no prior arrangement
    must still get a full comparison and exit 0; turning the simple flow into
    an error would make the feature unusable for the case it was built for.
    What changes is that the output says what it does not show, in the same
    place and voice `same_device` is called weaker than `independent`."""
    a, b = _honest_pair()
    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED
    assert r["commitment"]["both_verified"] is False
    assert r["provenance"]["commitments_verified"] is False
    text = va.format_compare(r)
    assert _NO_COMMITMENT_WORDS in text, text
    assert "WEAKER for the same reason two documents" in text
    assert "could have pasted its cell values into a document" in text
    # the qualifier must also ride on the RESULT line, because a reader who
    # greps for `RESULT:` or reads the last lines would otherwise get the
    # strong sentence with none of the reason it is weaker
    tail = text[text.index("RESULT: AGREE"):]
    assert "WEAKER THAN IT LOOKS" in tail, tail
    assert "python -m mojolearn verify --commitment" in text


def test_a_verified_pair_says_so_on_the_result_line():
    """UPDATED 2026-09-20, lane/compare-challenge-nonce, and the update is the
    finding. This used to assert `WEAKER THAN IT LOOKS` was absent from a
    commitment-verified pair. It is not absent any more, because a verified
    commitment answers ORDER and never answered EXECUTION: our reference table
    pins every expected cell hash, so both of these documents could have been
    written out of it and committed to without running. The commitment half of
    the result line is unqualified, as it always was; the challenge half is
    qualified, because this pair answers no challenge."""
    a, b = _honest_pair()
    la, lb = va.seal_document(a), va.seal_document(b)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED
    assert r["commitment"]["both_verified"] is True
    assert r["provenance"]["commitments_verified"] is True
    text = va.format_compare(r)
    assert la in text and lb in text, "both published commitments must be printed"
    tail = text[text.index("RESULT: AGREE"):]
    assert "neither set of numbers could have been copied" in tail, tail
    assert "WEAKER THAN IT LOOKS: no commitment was exchanged" not in tail, tail
    assert "WEAKER THAN IT LOOKS: no challenge was answered" in tail, tail


def test_one_sided_commitment_names_the_party_that_is_not_bound():
    """A one-sided exchange binds one party and leaves the other free to have
    copied. Saying only "commitments: partial" would hide which one."""
    a, b = _honest_pair()
    la = va.seal_document(a)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la)
    assert r["verdict"] == "AGREE"
    assert r["commitment"]["both_verified"] is False
    assert r["commitment"]["published_by"] == ["a.json"]
    text = va.format_compare(r)
    assert "only a.json is bound" in text, text
    assert "none was published for b.json" in text, text
    assert "stronger than none" in text and "much weaker than two" in text


def test_a_commitment_that_ships_inside_the_document_proves_nothing_and_says_so():
    """The reveal block carries both halves, so anyone holding the document can
    recompute it. Reporting that as a check that passed would be the purest
    form of a verification that cannot fail."""
    a, b = _honest_pair()
    va.seal_document(a)
    va.seal_document(b)
    r = va.compare_documents(a, b, "a.json", "b.json")   # neither line published
    assert r["commitment"]["a"]["state"] == "self-declared"
    assert r["commitment"]["both_verified"] is False
    text = va.format_compare(r)
    assert _NO_COMMITMENT_WORDS in text, text
    assert "travels with its own nonce" in text and "proves nothing" in text, text
    assert "WEAKER THAN IT LOOKS" in text


# ---------------------------------------------------------- the other moves

def test_one_commitment_presented_twice_is_refused():
    """The forger copies the document AND the line published for it. Both
    sides would then recompute to the same value and read `verified`."""
    a, _ = _honest_pair()
    line = va.seal_document(a)
    b = copy.deepcopy(a)
    b["elapsed_s"] = 3.0                       # not byte-identical any more
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=line, commitment_b=line)
    assert r["verdict"] == "COMMITMENT BROKEN" and r["exit"] == va.EXIT_MISMATCH
    assert any("SAME published commitment" in m for m in r["commitment"]["problems"]), \
        r["commitment"]["problems"]


def test_a_copied_nonce_is_refused():
    a, b = _honest_pair()
    la = va.seal_document(a)
    lb = va.seal_document(b, nonce=a[va.REVEAL_KEY]["nonce"])
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert any("same nonce" in m for m in r["commitment"]["problems"]), r["commitment"]["problems"]


def test_a_commitment_with_no_nonce_to_check_it_against_is_not_a_pass():
    """A published commitment and an unsealed document cannot be checked
    against each other. An unverifiable commitment must not read as a verified
    one, which is the shape of every defect this file is a list of."""
    a, b = _honest_pair()
    la = va.seal_document(a)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b="f" * 64)
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert r["commitment"]["b"]["state"] == "UNSEALED"
    assert "carries no `commitment_reveal` nonce" in va.format_compare(r)


def test_an_unreadable_commitment_is_not_silently_ignored():
    a, b = _honest_pair()
    va.seal_document(a)
    va.seal_document(b)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a="not-a-commitment")
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert r["commitment"]["a"]["state"] == "UNREADABLE"


def test_a_document_edited_after_sealing_is_reported_even_with_no_exchange():
    """Free catch, and never reported as anything milder. A forger who reseals
    defeats it, which is exactly why the PUBLISHED line is the mechanism and
    this is not."""
    a, b = _honest_pair()
    va.seal_document(a)
    a["cells"][0]["value"] = "ffff0000ffff0000"
    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert r["commitment"]["a"]["state"] == "SELF-INCONSISTENT"
    assert "edited after sealing" in va.format_compare(r)


def test_sealing_does_not_disarm_the_byte_identical_refusal():
    """A REGRESSION THIS FEATURE WOULD OTHERWISE CAUSE. `same_document`
    refuses one document handed over twice. A nonce is random per seal, so a
    naive implementation would make a sealed copy stop being byte-identical
    and quietly retire that refusal."""
    a, _ = _honest_pair()
    va.seal_document(a)
    copy_of_a = copy.deepcopy(a)
    va.seal_document(copy_of_a)                        # a fresh, different nonce
    assert copy_of_a[va.REVEAL_KEY]["nonce"] != a[va.REVEAL_KEY]["nonce"]
    r = va.compare_documents(a, copy_of_a, "a.json", "copy.json")
    assert r["verdict"] == "SAME DOCUMENT" and r["exit"] == va.EXIT_CANNOT_RUN, r["verdict"]


def test_a_broken_commitment_outranks_every_cell_outcome():
    """If a document is not the one its party committed to, the reader does
    not yet know that its cells are the cells that were computed, so no
    headline about those cells is honest. Only MALFORMED outranks it."""
    a, b = _honest_pair()
    la = va.seal_document(a)
    b["cells"][1]["value"] = "ffff0000ffff0000"        # a real MISMATCH as well
    va.seal_document(b)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a="0" * 64, commitment_b="1" * 64)
    assert r["differ"] == 1, "the cells really do disagree too"
    assert r["verdict"] == "COMMITMENT BROKEN"
    assert la not in ("0" * 64,)
    # and a document that will not parse still outranks it, and still reports it
    junk = dict(format="something-else", cells=[])
    r2 = va.compare_documents(junk, b, "junk.json", "b.json", commitment_b="1" * 64)
    assert r2["verdict"] == "MALFORMED" and r2["exit"] == va.EXIT_USAGE
    assert "COMMITMENTS: BROKEN" in va.format_compare(r2), va.format_compare(r2)


def test_two_seals_of_the_same_document_give_different_commitments():
    """The nonce is what stops a published commitment from publishing the
    document. Our reference table pins every expected cell hash, so a
    nonce-free hash of a predictable document could be brute-forced by
    whoever received it first."""
    doc = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    lines = set()
    for _ in range(5):
        doc.pop(va.REVEAL_KEY, None)
        lines.add(va.seal_document(doc))
    assert len(lines) == 5
    assert va.NONCE_BYTES * 8 >= 128


def test_a_commitment_reads_from_a_pasted_line_or_a_file(tmp_path):
    """64 characters get pasted into a message far more often than they get
    sent as a file. A tool that only accepts a file pushes people into writing
    the file themselves, badly."""
    doc = _doc("cuda", "RTX 4090", "nvidia", _CELLS)
    line = va.seal_document(doc)
    as_json = tmp_path / "x.commitment"
    as_json.write_text(json.dumps(dict(format=va.COMMITMENT_FORMAT, commitment=line)))
    as_text = tmp_path / "x.txt"
    as_text.write_text(line + "\n")
    for form in (line, str(as_json), str(as_text)):
        got, problem = va.read_published_commitment(form)
        assert problem is None and got == line, (form, problem)
    for bad in ("", "zz" * 32, str(tmp_path / "absent"), "0" * 63):
        got, problem = va.read_published_commitment(bad)
        assert got is None and problem, bad
    wrong_format = tmp_path / "y.commitment"
    wrong_format.write_text(json.dumps(dict(format="other", commitment=line)))
    got, problem = va.read_published_commitment(str(wrong_format))
    assert got is None and "expected" in problem


# ------------------------------------------------------------------ the CLI

def _run_cli(argv, env_extra=None):
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "mojolearn"] + argv, capture_output=True,
                          text=True, env=env, cwd=str(PKG.parent))


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return str(p)


def test_cli_seals_a_document_and_prints_one_line_to_publish(tmp_path):
    path = _write(tmp_path, "mine.json", _doc("cuda", "RTX 4090", "nvidia", _CELLS))
    r = _run_cli(["verify", "--commitment", path])
    assert r.returncode == va.EXIT_VERIFIED, r.stdout[-2000:] + r.stderr[-2000:]
    assert "RESULT: SEALED" in r.stdout
    hexes = re.findall(r"\b[0-9a-f]{64}\b", r.stdout)
    assert len(set(hexes)) == 1, r.stdout
    line = hexes[0]
    sealed = json.loads(Path(path).read_text())
    assert sealed[va.REVEAL_KEY]["commitment"] == line
    assert va.commitment_digest(sealed, sealed[va.REVEAL_KEY]["nonce"]) == line
    sidecar = json.loads(Path(path + ".commitment").read_text())
    assert sidecar["format"] == va.COMMITMENT_FORMAT and sidecar["commitment"] == line
    # THE NONCE MUST NOT BE IN THE THING YOU PUBLISH. Publishing it with the
    # commitment is publishing the document, which is the reveal, out of order.
    assert sealed[va.REVEAL_KEY]["nonce"] not in json.dumps(sidecar)
    assert sealed[va.REVEAL_KEY]["nonce"] not in r.stdout
    # idempotent: running it twice must not invalidate a published line
    again = _run_cli(["verify", "--commitment", path])
    assert again.returncode == va.EXIT_VERIFIED
    assert line in again.stdout


def test_cli_refuses_to_reprint_a_line_for_a_document_edited_after_sealing(tmp_path):
    path = _write(tmp_path, "mine.json", _doc("cuda", "RTX 4090", "nvidia", _CELLS))
    ok = _run_cli(["verify", "--commitment", path])
    assert ok.returncode == 0
    doc = json.loads(Path(path).read_text())
    doc["cells"][0]["value"] = "ffff0000ffff0000"
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    bad = _run_cli(["verify", "--commitment", path])
    assert bad.returncode == va.EXIT_MISMATCH, bad.stdout[-2000:]
    assert "RESULT: BROKEN" in bad.stdout
    assert "edited after it was sealed" in bad.stdout


def test_cli_the_whole_protocol_end_to_end(tmp_path):
    """THE EXIT CODE IS THE INTERFACE, through the real process, under
    `MOJOLEARN_NUMERIC_MODE=fast` as well: a third party has none of our
    bindings, and both `--commitment` and `--compare` must dispatch before the
    tier gate that refuses everything else with exit 3."""
    honest = _doc("metal", "Apple M2", "apple", _CELLS, commit="aaa")
    forger_own = _doc("cuda", "RTX 4090", "nvidia", _CELLS, commit="bbb")
    pa = _write(tmp_path, "mine.json", honest)
    pb = _write(tmp_path, "theirs.json", forger_own)

    for tier in ("identical", "fast"):
        for p in (pa, pb):
            doc = json.loads(Path(p).read_text())
            doc.pop(va.REVEAL_KEY, None)
            Path(p).write_text(json.dumps(doc, indent=1, sort_keys=True))
        # 1. both parties commit, before either sees the other's file
        ra = _run_cli(["verify", "--commitment", pa], dict(MOJOLEARN_NUMERIC_MODE=tier))
        rb = _run_cli(["verify", "--commitment", pb], dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert (ra.returncode, rb.returncode) == (0, 0), ra.stdout[-800:] + rb.stdout[-800:]
        la = json.loads(Path(pa + ".commitment").read_text())["commitment"]
        lb = json.loads(Path(pb + ".commitment").read_text())["commitment"]

        # 2. honest exchange
        ok = _run_cli(["verify", "--compare", pa, pb, "--commitment-a", la,
                       "--commitment-b", lb], dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert ok.returncode == va.EXIT_VERIFIED, ok.stdout[-2000:] + ok.stderr[-2000:]
        assert "RESULT: AGREE" in ok.stdout
        # the commitment qualifier is gone; the challenge one is not, because
        # these documents answer no challenge (lane/compare-challenge-nonce)
        assert "WEAKER THAN IT LOOKS: no commitment was exchanged" not in ok.stdout
        assert "WEAKER THAN IT LOOKS: no challenge was answered" in ok.stdout

        # 3. the forgery, handed over instead
        forged = _forged_from(json.loads(Path(pa).read_text()))
        pf = _write(tmp_path, "forged.json", forged)
        bad = _run_cli(["verify", "--compare", pa, pf, "--commitment-a", la,
                        "--commitment-b", lb], dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert bad.returncode == va.EXIT_MISMATCH, bad.stdout[-2000:]
        assert "RESULT: COMMITMENT BROKEN" in bad.stdout, bad.stdout[-2000:]
        assert "RESULT: AGREE" not in bad.stdout
        assert lb in bad.stdout, "the published line must be printed beside the one it is not"

        # 4. the same forgery with no commitments exchanged: AGREE, labelled
        weak = _run_cli(["verify", "--compare", pa, pf], dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert weak.returncode == va.EXIT_VERIFIED, weak.stdout[-2000:]
        assert _NO_COMMITMENT_WORDS in weak.stdout, weak.stdout[-2000:]
        assert "WEAKER THAN IT LOOKS" in weak.stdout

        # 5. the .commitment file is accepted where the line is
        viafile = _run_cli(["verify", "--compare", pa, pb, "--commitment-a", pa + ".commitment",
                            "--commitment-b", pb + ".commitment"],
                           dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert viafile.returncode == va.EXIT_VERIFIED, viafile.stdout[-2000:]

        # 6. swapping the two lines binds each document to the other's, and fails
        swapped = _run_cli(["verify", "--compare", pa, pb, "--commitment-a", lb,
                            "--commitment-b", la], dict(MOJOLEARN_NUMERIC_MODE=tier))
        assert swapped.returncode == va.EXIT_MISMATCH, swapped.stdout[-2000:]
        assert "RESULT: COMMITMENT BROKEN" in swapped.stdout


def test_cli_json_carries_the_commitment_block(tmp_path):
    honest = _doc("metal", "Apple M2", "apple", _CELLS, commit="aaa")
    theirs = _doc("cuda", "RTX 4090", "nvidia", _CELLS, commit="bbb")
    pa, pb = _write(tmp_path, "a.json", honest), _write(tmp_path, "b.json", theirs)
    la = json.loads(_run_cli(["verify", "--commitment", pa, "--json"]).stdout)["commitment"]
    lb = json.loads(_run_cli(["verify", "--commitment", pb, "--json"]).stdout)["commitment"]
    r = _run_cli(["verify", "--compare", pa, pb, "--commitment-a", la, "--commitment-b", lb,
                  "--json"])
    assert r.returncode == 0, r.stdout[-2000:]
    out = json.loads(r.stdout)
    assert out["commitment"]["both_verified"] is True
    assert out["commitment"]["a"]["state"] == "verified"
    assert out["provenance"]["commitments_verified"] is True
    plain = _run_cli(["verify", "--compare", pa, pb, "--json"])
    assert json.loads(plain.stdout)["commitment"]["both_verified"] is False
    assert plain.returncode == 0, "no commitment is a weaker result, never a failure"


def test_cli_rejects_the_flags_where_they_would_mean_nothing(tmp_path):
    path = _write(tmp_path, "mine.json", _doc("cuda", "RTX 4090", "nvidia", _CELLS))
    r = _run_cli(["verify", "--commitment-a", "0" * 64, "--all", "--quick"])
    assert r.returncode == va.EXIT_USAGE, r.stdout[-500:] + r.stderr[-500:]
    assert "only meaningful with --compare" in r.stderr
    r2 = _run_cli(["verify", "--commitment", path, "--compare", path, path])
    assert r2.returncode == va.EXIT_USAGE
    junk = tmp_path / "junk.json"
    junk.write_text("not json")
    r3 = _run_cli(["verify", "--commitment", str(junk)])
    assert r3.returncode == va.EXIT_USAGE and "RESULT: CANNOT READ" in r3.stdout
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps(dict(format="something-else", cells=[])))
    r4 = _run_cli(["verify", "--commitment", str(foreign)])
    assert r4.returncode == va.EXIT_USAGE and "NOT AN EVIDENCE DOCUMENT" in r4.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
