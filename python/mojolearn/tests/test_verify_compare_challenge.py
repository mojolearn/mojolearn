# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`verify --challenge` and `verify --compare --challenge-commitment-a/-b`:
the hole commit-reveal leaves open, and the challenge that closes it.

    cd python && python3 -m mojolearn.tests.test_verify_compare_challenge

A commitment settles the ORDER of two documents and nothing else. It cannot
say a document came from a run, because `mojolearn/verify_reference/table.json`
ships in the wheel with the expected hash of every cell in it: a party can
write a complete, well formed, contract-consistent evidence document straight
out of that table, seal it, publish the commitment first, and hand over a file
that never executed a line.

THAT ATTACK IS BUILT HERE FIRST, out of the real shipped table, and WATCHED TO
PASS AS `AGREE` WITH BOTH COMMITMENTS VERIFIED -- because a defence whose
failure was never observed is not a defence -- and only then run again against
the challenge and watched to be caught.

Nothing here needs a GPU, a binding or a network. The one test that exercises
the challenge RUNNER drives it through a stub harness written to a temporary
file, so the ordering and the plumbing are watched without fitting anything.
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

from mojolearn import _verify_all as va
from mojolearn import _verify_reference as vref

PKG = Path(va.__file__).resolve().parent


# --------------------------------------------------------- the real synthesis

#: A handful of lanes and fixtures. The attack does not get stronger with
#: more cells and the test does not get truer; what matters is that every
#: value in it came out of the shipped table rather than out of a machine.
_LANES = ("ols", "kmeans", "pca", "rf-clf")
_FIXTURES = ("base", "ties")

#: What `verification_contract` records for the parts a comparison checks a
#: protocol for. A SYNTHESIST HAS THESE TOO: they are built by importing the
#: harness and reading its constants, with nothing fitted, so pretending the
#: forger could not produce them would be building a weaker attack than the
#: real one.
_PROTOCOLS = {"batch": {"alone": 16, "split": [2, 4, "n"], "prefix": "1,7,full-1", "enabled": True},
              "stepfull": {"version": 1, "positions": "all"}}


def _table():
    path = vref.table_path()
    if not os.path.exists(path):
        pytest.skip(f"no reference table at {path}; the synthesis attack is built from it")
    return vref.load_table(path)


def _synthesized(table, vendor, device, device_class, commit):
    """THE ATTACK. No estimator is constructed, no fit is run, no binding is
    loaded: every hash below is read out of the table that ships in the wheel,
    and the provenance block is typed.

    This is the document `docs/VERIFY_EXTERNALLY.md` said a commitment cannot
    rule out, written out so a reader does not have to imagine it."""
    rows = []
    for lane in _LANES:
        for fixture in _FIXTURES:
            for part in vref.PARTS:
                ent = vref.entry(table, lane, fixture, part)
                if ent is None or ent.get("ref") is None:
                    continue
                rows.append(dict(lane=lane, fixture=fixture, part=part,
                                 value=ent["ref"], state=vref.IDENTICAL))
    if not rows:
        pytest.skip("the shipped table carries no reference for these lanes")
    contract = dict(
        harness_sha256=table.get("harness_sha256"),
        fixtures={f: table["fixtures"][f] for f in _FIXTURES if f in table.get("fixtures", {})},
        heldout={f: table["heldout"][f] for f in _FIXTURES if f in table.get("heldout", {})},
        protocols=dict(_PROTOCOLS))
    return dict(
        format=va.COMPARE_INPUT_FORMAT, verdict="VERIFIED",
        detail=f"verified {len(rows)} of {len(rows)} cell parts",
        elapsed_s=1481.2, lanes=list(_LANES),
        lane_seconds={l: 42.0 for l in _LANES},
        device=dict(mojolearn_version="0.8.7", commit=commit, commit_source="witness",
                    numeric_mode="identical", vendor=vendor, device_class=device_class,
                    device=device, cpu_model="cpu-x", requested_parallel_devices=[],
                    platform="p", python="3.14.6", numpy="2.3.1"),
        bindings=[dict(module="mojolearn._mojolearn", sha256="d" * 64, size=1)],
        verification_contract=contract, cells=rows)


def _pair_from_the_table():
    """Two documents, neither of which ran anything, claiming two vendors."""
    table = _table()
    return (_synthesized(table, "metal", "Apple M4", "apple", "aaa111222"),
            _synthesized(table, "cuda", "RTX 4090", "nvidia", "bbb333444"))


# ------------------------------------------------ the hole, demonstrated first

def test_two_documents_written_out_of_the_shipped_table_read_agree():
    """WATCH IT PASS FIRST. Neither document came from a run. Both are well
    formed, their contracts match, no cell is MOVED, none is one-sided, the
    files are not byte-identical, and both parties committed before the
    exchange -- so every defence that existed before this lane is satisfied
    and the comparer says AGREE across two "independent" vendors.

    This test asserts the HOLE, not a fix. If `--compare` ever learns to
    reject a synthesized document some other way, that is a real finding and
    this test should be rewritten to say so; it must never be deleted on the
    assumption that the challenge covers it, because the challenge only fires
    when a challenge was actually exchanged."""
    a, b = _pair_from_the_table()
    la, lb = va.seal_document(a), va.seal_document(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json",
                             commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED, r["verdict"]
    assert r["differ"] == 0 and r["agree"] > 0
    assert r["commitment"]["both_verified"] is True, "and both commitments VERIFY"
    assert r["provenance"]["independent"] is True, (
        "the synthesis even reads as two independent hardware classes, which is the strongest "
        "thing this command can say about a pair of documents")
    text = va.format_compare(r)
    assert "RESULT: AGREE" in text
    # ...and the one thing that is now said about it, on the RESULT line
    tail = text[text.index("RESULT: AGREE"):]
    assert "WEAKER THAN IT LOOKS: no challenge was answered" in tail, tail


def test_the_synthesis_is_really_the_shipped_table_and_not_a_mock():
    """A SYNTHESIS ATTACK BUILT OUT OF MADE-UP HASHES WOULD PROVE NOTHING. The
    values above have to be the ones the wheel publishes, or the test is about
    an attack nobody can run."""
    table = _table()
    doc = _synthesized(table, "cuda", "RTX 4090", "nvidia", "bbb")
    checked = 0
    for row in doc["cells"]:
        ent = vref.entry(table, row["lane"], row["fixture"], row["part"])
        assert row["value"] == ent["ref"], (row, ent)
        checked += 1
    assert checked >= 8, f"only {checked} cells came from the table"


# --------------------------------------------------- the challenge, and the fix

def _challenge_block(challenge, derived_from, salt="honest", lanes=_LANES):
    """One party's challenge response.

    The VALUES are a stand-in for what the challenge runner computes; what
    matters for every test below is that they are a function of the challenge,
    which is what makes them unwritable in advance, and equal between two
    honest parties, which is what makes them comparable."""
    import hashlib
    def h(*parts):
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    cells = [dict(lane=lane, part=part, value=h(challenge, lane, part, salt), error=None)
             for lane in lanes for part in vref.PARTS]
    return dict(format=va.CHALLENGE_FORMAT, challenge=challenge,
                derived_from=sorted(derived_from),
                fixture=dict(kind=va.CHALLENGE_FIXTURE_KIND, n=20000, d=16,
                             train_seed=str(va.challenge_seeds(challenge)[0]),
                             heldout_seed=str(va.challenge_seeds(challenge)[1]),
                             X=h(challenge, "X"), y_clf=h(challenge, "yc"),
                             y_reg=h(challenge, "yr"), heldout_X=h(challenge, "held")),
                harness_sha256="e" * 64, lanes=list(lanes), cells=cells, seconds=101.5)


def _challenged_pair(seal_challenges=True):
    """The whole protocol, in the order a party runs it:
    run, commit, exchange lines, answer, commit again, exchange documents."""
    a, b = _pair_from_the_table()
    la, lb = va.seal_document(a), va.seal_document(b)          # round one
    challenge, problem = va.derive_challenge(la, lb)           # after the exchange of lines
    assert problem is None, problem
    a[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb))
    b[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb))
    ca = cb = None
    if seal_challenges:
        ca, cb = va.seal_challenge(a), va.seal_challenge(b)    # round two
    return a, b, la, lb, ca, cb, challenge


def test_a_correct_pair_still_reads_agree_and_says_what_it_now_shows():
    a, b, la, lb, ca, cb, challenge = _challenged_pair()
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED, r
    assert r["challenge"]["both_verified"] is True
    assert r["provenance"]["challenge"] == challenge
    assert r["provenance"]["challenge_verified"] is True
    # the challenge cells are counted with the rest, not in a corner of their own
    assert r["agree"] == len(r["agreeing"]) and len(r["challenge"]["agree"]) > 0
    assert r["agree"] > len(_LANES) * len(_FIXTURES)
    text = va.format_compare(r)
    assert challenge in text, "the challenge itself must be printed, not summarized"
    tail = text[text.index("RESULT: AGREE"):]
    assert "WEAKER THAN IT LOOKS" not in tail, tail
    assert "neither document could have been written out of our" in tail, tail


def test_the_synthesist_has_no_challenge_to_answer_and_is_caught():
    """THE FIX, against the attack this lane exists for. The honest party
    answers the challenge. The synthesist cannot: the fixture is derived from
    two commitments that did not exist when they wrote their document, and no
    hash for it is in the table they copied from. What they hand over carries
    no challenge block, and a one-sided challenge cannot be checked at all."""
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=False)
    ca = va.seal_challenge(a)
    synthesist = copy.deepcopy(b)
    synthesist.pop(va.CHALLENGE_KEY)                 # they never ran anything
    r = va.compare_documents(a, synthesist, "honest.json", "synth.json",
                             commitment_a=la, commitment_b=lb, challenge_commitment_a=ca)
    assert r["verdict"] == "CHALLENGE BROKEN" and r["exit"] == va.EXIT_MISMATCH, r["verdict"]
    assert any("only honest.json answers a challenge" in m for m in r["challenge"]["problems"]), \
        r["challenge"]["problems"]
    text = va.format_compare(r)
    assert "RESULT: CHALLENGE BROKEN" in text and "RESULT: AGREE" not in text
    # and the sentence a skimmer would quote must not appear over it
    assert "two independent machines reaching the" not in text, text
    assert "PROVENANCE ABOVE CANNOT BE TAKEN AT FACE VALUE" in text, text


def test_neither_party_answering_is_legal_and_labelled_rather_than_failed():
    """BACKWARD COMPATIBILITY, and it is a design decision rather than an
    accident. Every document written by a release before this lane carries no
    challenge, and `bench/results/verify_reports/` publishes two of them. They
    must still compare, and they must be labelled as carrying the weaker
    guarantee, exactly as a comparison without commitments already is."""
    a, b = _pair_from_the_table()
    la, lb = va.seal_document(a), va.seal_document(b)
    r = va.compare_documents(a, b, "old-a.json", "old-b.json", commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED
    assert r["challenge"]["broken"] is False and r["challenge"]["challenge"] is None
    text = va.format_compare(r)
    assert "CHALLENGE: none." in text, text
    assert "written straight out of it" in text, text
    assert "WEAKER THAN IT LOOKS: no challenge was answered" in text
    assert "verify --challenge mine.json" in text, "and it must say how to close it"


def test_a_stale_challenge_from_an_earlier_seal_of_the_same_files_is_caught():
    """The sharpest of the three. The two parties really did exchange, really
    did answer, and then one of them RESEALED -- a fresh nonce, a new round-one
    commitment -- and handed over the old response. Everything about the
    response is internally consistent; it just belongs to a pairing that no
    longer exists."""
    a, b, la, lb, ca, cb, challenge = _challenged_pair()
    new_la = va.seal_document(a)                      # a fresh nonce: a new commitment
    assert new_la != la
    r = va.compare_documents(a, b, "apple.json", "nvidia.json",
                             commitment_a=new_la, commitment_b=lb)
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]
    assert any("different exchange" in m for m in r["challenge"]["problems"]), \
        r["challenge"]["problems"]


def test_a_challenge_from_a_different_pair_is_caught():
    """Two parties answer a challenge derived from somebody else's
    commitments -- an older exchange of their own, or a pair they were handed.
    The derivation is correct; the pair is not theirs."""
    a, b, la, lb, ca, cb, _c = _challenged_pair(seal_challenges=False)
    stranger_x, stranger_y = "1" * 64, "2" * 64
    other, problem = va.derive_challenge(stranger_x, stranger_y)
    assert problem is None
    for doc in (a, b):
        doc[va.CHALLENGE_KEY] = _challenge_block(other, (stranger_x, stranger_y))
    ca, cb = va.seal_challenge(a), va.seal_challenge(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]
    problems = " ".join(r["challenge"]["problems"])
    assert "not the pair of commitments these two" in problems, problems
    assert "not the pair of commitments published" in problems, problems


def test_a_self_chosen_challenge_is_caught():
    """The obvious move once a party understands what the challenge is for:
    pick a value you have already answered, and write two commitments beside
    it. The derivation is recomputed rather than believed, so the value and
    the pair have to agree with each other."""
    a, b, la, lb, _ca, _cb, challenge = _challenged_pair(seal_challenges=False)
    chosen = "c0ffee" + "0" * 58
    assert chosen != challenge
    for doc in (a, b):
        doc[va.CHALLENGE_KEY] = _challenge_block(chosen, (la, lb))
    ca, cb = va.seal_challenge(a), va.seal_challenge(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]
    assert any("is not a value a party" in m for m in r["challenge"]["problems"]), \
        r["challenge"]["problems"]


def test_one_party_committing_to_itself_twice_cannot_derive_a_challenge():
    """A party who plays both sides with ONE commitment gets no challenge at
    all: the derivation refuses an equal pair, because a challenge derived
    from one line handed over twice is a challenge one party chose alone."""
    value, problem = va.derive_challenge("a" * 64, "a" * 64)
    assert value is None and "presented twice" in problem, problem


def test_a_copied_challenge_response_is_caught_by_the_second_commitment():
    """WHY THE PROTOCOL GROWS A SECOND ROUND, held as a test.

    TWO HONEST RESPONSES ARE IDENTICAL -- that is the point of them -- so a
    copy is not something a comparer can see in the bytes. What the copier
    cannot do is publish a commitment over a response they do not have yet:
    they have to publish SOMETHING before the exchange, and whatever they
    guessed is not what they hand over afterwards.
    """
    a, b, la, lb, _ca, _cb, challenge = _challenged_pair(seal_challenges=False)
    ca = va.seal_challenge(a)
    # the copier never ran: before the exchange they can only commit to a
    # response they made up
    b[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb), salt="guessed")
    cb = va.seal_challenge(b)
    # ...then the honest file arrives and they paste its answers in
    b[va.CHALLENGE_KEY] = copy.deepcopy(a[va.CHALLENGE_KEY])
    r = va.compare_documents(a, b, "honest.json", "copier.json",
                             commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    # the cells agree perfectly; the verdict is not about the cells
    assert r["differ"] == 0
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]
    assert r["challenge"]["b"]["state"] in ("MISMATCH", "SELF-INCONSISTENT"), r["challenge"]["b"]
    # and had they handed over the guess instead, the challenge answers
    # themselves would have disagreed
    b[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb), salt="guessed")
    va.seal_challenge(b, nonce=b[va.REVEAL_KEY][va.CHALLENGE_NONCE_FIELD])
    r2 = va.compare_documents(a, b, "honest.json", "copier.json",
                              commitment_a=la, commitment_b=lb,
                              challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r2["verdict"] == "MISMATCH", r2["verdict"]


def test_without_the_second_round_a_copied_response_passes_and_says_so():
    """AND WATCH THAT GAP, rather than claiming the second round is optional.
    With no challenge commitments exchanged, a copied response is
    indistinguishable from an honest one, the comparison still AGREEs, and the
    output has to say what it does not show."""
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=False)
    b[va.CHALLENGE_KEY] = copy.deepcopy(a[va.CHALLENGE_KEY])
    r = va.compare_documents(a, b, "honest.json", "copier.json",
                             commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "AGREE" and r["challenge"]["both_verified"] is False
    text = va.format_compare(r)
    assert "WEAKER: neither response was committed to" in text, text
    assert "could have copied the challenge block" in text, text
    tail = text[text.index("RESULT: AGREE"):]
    assert "WEAKER THAN IT LOOKS: the challenge was answered but" in tail, tail


def test_a_challenge_answered_over_different_input_is_not_read_as_a_divergence():
    """THE WORST FAILURE THIS PROJECT HAS is reporting DIVERGENT for a reason
    that is not arithmetic. If two boxes somehow draw different bytes from one
    challenge, that must be named as different INPUT, not as 5,000 cells
    disagreeing."""
    a, b, la, lb, ca, cb, _c = _challenged_pair(seal_challenges=False)
    b[va.CHALLENGE_KEY]["fixture"]["X"] = "ffff0000ffff0000"
    ca, cb = va.seal_challenge(a), va.seal_challenge(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "CHALLENGE BROKEN"
    assert any("DIFFERENT input" in m for m in r["challenge"]["problems"]), \
        r["challenge"]["problems"]
    assert "answers to different questions" in va.format_compare(r)


def test_a_challenge_answer_that_differs_is_a_mismatch_not_a_broken_challenge():
    """A DIVERGENCE ON THE CHALLENGE FIXTURE IS THE STRONGEST FINDING THIS
    COMMAND CAN PRODUCE, and it must not be softened into a complaint about
    the protocol. The challenge held up; the machines disagreed."""
    a, b, la, lb, ca, cb, _c = _challenged_pair(seal_challenges=False)
    b[va.CHALLENGE_KEY]["cells"][0]["value"] = "ffff0000ffff0000"
    ca, cb = va.seal_challenge(a), va.seal_challenge(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "MISMATCH" and r["exit"] == va.EXIT_MISMATCH, r["verdict"]
    assert r["differ"] == 1
    named = va.format_compare(r)
    assert "challenge:" in named, "the differing challenge cell must be NAMED, not counted"


def test_a_narrowed_challenge_shows_up_as_cells_only_one_side_carries():
    """A party who answers fewer lanes than the other is not agreeing about
    the rest. `INCOMPLETE`, by the ladder that already exists."""
    a, b, la, lb, _ca, _cb, challenge = _challenged_pair(seal_challenges=False)
    b[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb), lanes=_LANES[:2])
    ca, cb = va.seal_challenge(a), va.seal_challenge(b)
    r = va.compare_documents(a, b, "apple.json", "nvidia.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "INCOMPLETE" and r["exit"] == va.EXIT_CANNOT_RUN, r["verdict"]
    assert len(r["only_in_a"]) > 0


# ------------------------------------------------------------ the ladder itself

def test_challenge_broken_sits_directly_under_commitment_broken():
    """A broken commitment outranks a broken challenge, because a document
    that is not the one its party committed to is not a document whose
    challenge is worth reading; and a file that will not parse outranks both."""
    a, b, la, lb, ca, cb, _c = _challenged_pair()
    b[va.CHALLENGE_KEY]["challenge"] = "d" * 64                  # challenge broken
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "CHALLENGE BROKEN"
    # ...and with the commitments broken too, the commitment wins
    r2 = va.compare_documents(a, b, "a.json", "b.json", commitment_a="0" * 64,
                              commitment_b="1" * 64,
                              challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r2["verdict"] == "COMMITMENT BROKEN", r2["verdict"]
    # ...and a file that will not parse outranks both, while still reporting them
    junk = dict(format="something-else", cells=[])
    r3 = va.compare_documents(junk, b, "junk.json", "b.json", challenge_commitment_b=cb)
    assert r3["verdict"] == "MALFORMED" and r3["exit"] == va.EXIT_USAGE
    assert "CHALLENGE: BROKEN" in va.format_compare(r3), va.format_compare(r3)


def test_a_broken_challenge_outranks_a_cell_mismatch():
    """Same reasoning as COMMITMENT BROKEN outranking MISMATCH: until the
    reader knows these cells were computed, no headline about the cells is
    honest."""
    a, b, la, lb, ca, cb, _c = _challenged_pair()
    b["cells"][0]["value"] = "ffff0000ffff0000"                 # a real MISMATCH too
    b[va.CHALLENGE_KEY]["derived_from"] = ["9" * 64, "8" * 64]  # and a broken challenge
    lb = va.seal_document(b)                                    # commitments consistent again
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb)
    assert r["commitment"]["broken"] is False, r["commitment"]["problems"]
    assert r["differ"] == 1, "the cells really do disagree too"
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]


# ------------------------------------------------------ what the digests cover

def test_the_challenge_is_a_function_of_both_commitments_and_of_neither_alone():
    la, lb, lc = "a" * 64, "b" * 64, "c" * 64
    ab, _ = va.derive_challenge(la, lb)
    ba, _ = va.derive_challenge(lb, la)
    ac, _ = va.derive_challenge(la, lc)
    assert ab == ba, "sorted, so neither party has to be A"
    assert ab != ac, "it moves when either commitment moves"
    assert len(ab) == 64


def test_the_challenge_domain_is_not_the_commitment_domain():
    """DOMAIN SEPARATION, the same rule `_COMMITMENT_DOMAIN` states. A
    challenge is 64 hex characters printed in the same kind of message a
    commitment is, so a shared domain would let a value lifted out of one slot
    be replayed in the other."""
    assert va._CHALLENGE_DOMAIN != va._COMMITMENT_DOMAIN
    assert va._CHALLENGE_COMMITMENT_DOMAIN not in (va._COMMITMENT_DOMAIN, va._CHALLENGE_DOMAIN)
    doc = _pair_from_the_table()[0]
    line = va.seal_document(doc)
    nonce = doc[va.REVEAL_KEY]["nonce"]
    doc[va.CHALLENGE_KEY] = _challenge_block(*va.derive_challenge(line, "b" * 64)[:1],
                                             derived_from=(line, "b" * 64))
    assert va.challenge_commitment_digest(doc, nonce) != va.commitment_digest(doc, nonce)


def test_adding_a_challenge_does_not_break_an_already_published_commitment():
    """THE COMPATIBILITY THAT MAKES THE PROTOCOL POSSIBLE. The challenge is
    derived from the round-one commitment, so the response necessarily arrives
    after that line is published. If answering moved the line, the check would
    fire on the protocol working."""
    a, b = _pair_from_the_table()
    la, lb = va.seal_document(a), va.seal_document(b)
    challenge, _ = va.derive_challenge(la, lb)
    a[va.CHALLENGE_KEY] = _challenge_block(challenge, (la, lb))
    assert va.commitment_digest(a, a[va.REVEAL_KEY]["nonce"]) == la
    assert va.commitment_state(a, la, "a.json")["state"] == "verified"
    assert "challenge" in va.COMMITMENT_EXCLUDES


_CHALLENGE_COVERED = [(("format",), "other"),
                      (("challenge",), "f" * 64),
                      (("derived_from",), ["9" * 64, "8" * 64]),
                      (("fixture", "X"), "ffff0000ffff0000"),
                      (("harness_sha256",), "f" * 64),
                      (("lanes",), ["ols"]),
                      (("cells", 0, "value"), "ffff0000ffff0000"),
                      (("cells", 0, "lane"), "renamed"),
                      (("cells", 0, "part"), "renamed")]


def test_every_field_the_comparer_reads_moves_the_challenge_commitment():
    """The same rule `commitment_preimage` is held to: cover exactly the
    fields the comparison reads. A field read and not covered is a field a
    party may still edit after the exchange."""
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=False)
    nonce = "ab" * 16
    ref = va.challenge_commitment_digest(a, nonce)
    uncovered = []
    for path, value in _CHALLENGE_COVERED:
        doc = copy.deepcopy(a)
        node = doc[va.CHALLENGE_KEY]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        if va.challenge_commitment_digest(doc, nonce) == ref:
            uncovered.append("/".join(map(str, path)))
    assert not uncovered, ("read by the comparer and NOT covered by the challenge commitment: "
                           + "; ".join(uncovered))
    # and the wall clock is not covered, because no comparison reads it
    doc = copy.deepcopy(a)
    doc[va.CHALLENGE_KEY]["seconds"] = 9999.0
    assert va.challenge_commitment_digest(doc, nonce) == ref
    assert set(va.CHALLENGE_EXCLUDES) == {"seconds"}


def test_the_covered_list_is_pinned_so_a_change_has_to_be_deliberate():
    assert va.CHALLENGE_COVERS == ("format", "challenge", "derived_from", "fixture",
                                   "harness_sha256", "lanes", "cells")


def test_the_challenge_commitment_is_bound_to_the_document_it_answers_for():
    """A second line lifted off one document and presented for another must
    not verify. The round-one commitment is in the preimage for exactly that."""
    a, b, la, lb, ca, cb, _c = _challenged_pair()
    assert ca != cb
    swapped = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb,
                                   challenge_commitment_a=cb, challenge_commitment_b=ca)
    assert swapped["verdict"] == "CHALLENGE BROKEN", swapped["verdict"]
    assert swapped["challenge"]["a"]["state"] == "MISMATCH"


def test_a_challenge_commitment_that_ships_inside_the_document_proves_nothing():
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=True)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb)
    assert r["challenge"]["a"]["state"] == "self-declared"
    assert r["challenge"]["both_verified"] is False
    assert r["verdict"] == "AGREE", "self-declared is a weaker result, never a failure"


def test_a_copied_challenge_nonce_is_refused():
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=False)
    ca = va.seal_challenge(a)
    cb = va.seal_challenge(b, nonce=a[va.REVEAL_KEY][va.CHALLENGE_NONCE_FIELD])
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a=ca, challenge_commitment_b=cb)
    assert r["verdict"] == "CHALLENGE BROKEN"
    assert any("same challenge nonce" in m for m in r["challenge"]["problems"]), \
        r["challenge"]["problems"]


def test_a_published_challenge_commitment_with_no_response_to_check_is_not_a_pass():
    a, b = _pair_from_the_table()
    la, lb = va.seal_document(a), va.seal_document(b)
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb,
                             challenge_commitment_a="f" * 64)
    assert r["verdict"] == "CHALLENGE BROKEN", r["verdict"]
    assert any("never run" in m for m in r["challenge"]["problems"]), r["challenge"]["problems"]


def test_a_malformed_challenge_block_is_broken_rather_than_ignored():
    a, b, la, lb, _ca, _cb, _c = _challenged_pair(seal_challenges=False)
    b[va.CHALLENGE_KEY]["cells"].append(dict(b[va.CHALLENGE_KEY]["cells"][0]))
    r = va.compare_documents(a, b, "a.json", "b.json", commitment_a=la, commitment_b=lb)
    assert r["verdict"] == "CHALLENGE BROKEN"
    assert any("appears twice" in m for m in r["challenge"]["problems"]), r["challenge"]["problems"]


# ------------------------------------------- the runner, through a stub harness

_STUB_HARNESS = '''
"""A harness stand-in: no numpy, no bindings, no fitting. Every value is a
function of the fixture seed, which is what the challenge moves."""
import hashlib

BATCH_ALONE = 16
FIXTURES = ["base", "hashed"]


class Arr:
    def __init__(self, token, n=20000, d=16):
        self.token, self.shape = token, (n, d)

    def copy(self):
        return Arr(self.token, *self.shape)


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        m.update(str(getattr(a, "token", a)).encode())
    return m.hexdigest()[:16]


def fixture(kind, n=20000, d=16, seed=0):
    return (Arr(f"{kind}:{seed}:X", n, d), Arr(f"{kind}:{seed}:yc", n, 1),
            Arr(f"{kind}:{seed}:yr", n, 1))


def heldout(kind):
    return fixture(kind, seed=1)[0]


def _train_hash(parts):
    return hashlib.sha256("|".join(f"{k}={v}" for k, v in sorted(parts.items())).encode()).hexdigest()[:16]


def _lane(name):
    def run(ml, X, yc, yr, held):
        return dict(lane=name, w=_h(X), y=_h(yc))
    return run


LANES = {n: _lane(n) for n in ("ols", "kmeans", "pca")}
EXTRA_PARTS = {"stepfull": "decode"}


def _probe_fit(fit, name):
    base = _train_hash(fit)
    return _h(base + ":infer"), _h(base + ":model"), _h(base + ":infer"), None


def _probe_batch(fit, name, ml, Xh, alone, sabotage):
    return _h(_train_hash(fit) + ":batch" + str(alone)), None


def _probe_part(part, fit, name, ml, Xh, alone, tag):
    return _h(_train_hash(fit) + ":" + part), None, {}
'''


class _Args:
    def __init__(self, **kw):
        self.json = False
        self.lanes = ""
        self.repeats = 1
        self.challenge = None
        self.challenge_from = None
        self.__dict__.update(kw)


@pytest.fixture()
def stub_harness(tmp_path, monkeypatch):
    path = tmp_path / "stub_harness.py"
    path.write_text(_STUB_HARNESS, encoding="utf-8")
    monkeypatch.setenv("MOJOLEARN_IDENTITY_BREAK", str(path))
    sys.modules.pop("mojolearn_verify_all_harness", None)
    yield str(path)
    sys.modules.pop("mojolearn_verify_all_harness", None)


def _runnable_doc(harness_sha, lanes=("ols", "kmeans", "pca")):
    doc = _synthesized(_table(), "cuda", "RTX 4090", "nvidia", "bbb")
    doc["lanes"] = list(lanes)
    doc["verification_contract"]["harness_sha256"] = harness_sha
    return doc


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return str(p)


def test_the_runner_answers_and_the_answer_moves_with_the_challenge(tmp_path, stub_harness):
    """THE PROPERTY THE WHOLE LANE RESTS ON, watched rather than argued: the
    hashes a run produces are a function of the challenge, so a document
    written before the challenge existed cannot carry them."""
    sha = vref.sha256_file(stub_harness)
    answers = {}
    for tag, (x, y) in dict(one=("a" * 64, "b" * 64), two=("a" * 64, "c" * 64)).items():
        doc = _runnable_doc(sha)
        path = _write(tmp_path, f"{tag}.json", doc)
        doc = json.loads(Path(path).read_text())
        va.seal_document(doc)
        Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
        line = doc[va.REVEAL_KEY]["commitment"]
        code = va._cmd_challenge(_Args(challenge=path, challenge_from=[line, y]), ml=None)
        assert code == va.EXIT_VERIFIED, code
        answered = json.loads(Path(path).read_text())[va.CHALLENGE_KEY]
        answers[tag] = answered
    assert answers["one"]["challenge"] != answers["two"]["challenge"]
    assert answers["one"]["fixture"]["X"] != answers["two"]["fixture"]["X"], (
        "the challenge must move the INPUT, or it is not mixed into what the run computes")
    va_one = {(c["lane"], c["part"]): c["value"] for c in answers["one"]["cells"]}
    va_two = {(c["lane"], c["part"]): c["value"] for c in answers["two"]["cells"]}
    assert set(va_one) == set(va_two) and va_one, "the same cells, answered twice"
    assert all(va_one[k] != va_two[k] for k in va_one), (
        "every answer must move with the challenge; an answer that does not is an answer that "
        "could have been written in advance")


def test_the_runner_refuses_a_document_the_challenge_is_not_about(tmp_path, stub_harness):
    sha = vref.sha256_file(stub_harness)
    doc = _runnable_doc(sha)
    path = _write(tmp_path, "mine.json", doc)
    doc = json.loads(Path(path).read_text())
    va.seal_document(doc)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    code = va._cmd_challenge(_Args(challenge=path, challenge_from=["a" * 64, "b" * 64]), ml=None)
    assert code == va.EXIT_USAGE, code
    assert va.CHALLENGE_KEY not in json.loads(Path(path).read_text())


def test_the_runner_refuses_an_unsealed_document(tmp_path, stub_harness):
    path = _write(tmp_path, "mine.json", _runnable_doc(vref.sha256_file(stub_harness)))
    code = va._cmd_challenge(_Args(challenge=path, challenge_from=["a" * 64, "b" * 64]), ml=None)
    assert code == va.EXIT_USAGE


def test_the_runner_refuses_to_answer_a_second_challenge(tmp_path, stub_harness):
    """A party who could keep trying pairs until one suited them would be
    choosing the challenge after all."""
    sha = vref.sha256_file(stub_harness)
    path = _write(tmp_path, "mine.json", _runnable_doc(sha))
    doc = json.loads(Path(path).read_text())
    line = va.seal_document(doc)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True))
    assert va._cmd_challenge(_Args(challenge=path, challenge_from=[line, "b" * 64]),
                            ml=None) == va.EXIT_VERIFIED
    # the same pair again is idempotent and must not invalidate the published line
    before = json.loads(Path(path).read_text())[va.REVEAL_KEY][va.CHALLENGE_COMMITMENT_FIELD]
    assert va._cmd_challenge(_Args(challenge=path, challenge_from=[line, "b" * 64]),
                            ml=None) == va.EXIT_VERIFIED
    assert json.loads(Path(path).read_text())[va.REVEAL_KEY][va.CHALLENGE_COMMITMENT_FIELD] == before
    # a DIFFERENT pair is refused
    assert va._cmd_challenge(_Args(challenge=path, challenge_from=[line, "c" * 64]),
                            ml=None) == va.EXIT_USAGE


def test_two_stub_runs_of_the_same_challenge_agree_end_to_end(tmp_path, stub_harness):
    """The honest path through the whole protocol, with the responses actually
    computed by the runner rather than written by the test."""
    sha = vref.sha256_file(stub_harness)
    paths, lines = [], []
    for tag, vendor in (("a", "metal"), ("b", "cuda")):
        doc = _runnable_doc(sha)
        doc["device"].update(vendor=vendor, device_class=("apple" if tag == "a" else "nvidia"),
                             device=("Apple M4" if tag == "a" else "RTX 4090"))
        p = _write(tmp_path, f"{tag}.json", doc)
        doc = json.loads(Path(p).read_text())
        lines.append(va.seal_document(doc))
        Path(p).write_text(json.dumps(doc, indent=1, sort_keys=True))
        paths.append(p)
    seconds = []
    for p in paths:
        assert va._cmd_challenge(_Args(challenge=p, challenge_from=list(lines)),
                                ml=None) == va.EXIT_VERIFIED
        seconds.append(json.loads(Path(p).read_text())[va.CHALLENGE_KEY]["seconds"])
    docs = [json.loads(Path(p).read_text()) for p in paths]
    challenge_lines = [d[va.REVEAL_KEY][va.CHALLENGE_COMMITMENT_FIELD] for d in docs]
    assert challenge_lines[0] != challenge_lines[1], "different nonces, different second lines"
    r = va.compare_documents(docs[0], docs[1], "a.json", "b.json",
                             commitment_a=lines[0], commitment_b=lines[1],
                             challenge_commitment_a=challenge_lines[0],
                             challenge_commitment_b=challenge_lines[1])
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED, r["verdict"]
    assert r["challenge"]["both_verified"] is True
    assert len(r["challenge"]["agree"]) == 3 * len(vref.PARTS)
    text = va.format_compare(r)
    assert "neither document could have been written out of our" in text[text.index("RESULT: AGREE"):]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
