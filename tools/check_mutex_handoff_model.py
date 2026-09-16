#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enumerate two successful lock holders with fresh atomic observations.

This bounded model checks happens-before edges, not hardware timing. It is
deliberately stricter than C++ (all loads see the latest modification), yet
still exposes the old protocol. Failed/spurious attempts are omitted. The
general argument, including stale-load coherence, is in the review note.
"""
import argparse


def interleave(left, right, prefix=()):
    if not left and not right:
        yield prefix
    if left:
        yield from interleave(left[1:], right, prefix + (left[0],))
    if right:
        yield from interleave(left, right[1:], prefix + (right[0],))


def ordered(edges, start, end):
    seen, pending = set(), [start]
    while pending:
        here = pending.pop()
        if here == end:
            return True
        if here not in seen:
            seen.add(here)
            pending.extend(b for a, b in edges if a == here)
    return False


def check(protocol):
    operations = ("test", "cas", "payload", "release")
    if protocol == "postload":
        operations = ("test", "cas", "postload", "payload", "release")
    threads = [tuple((t, op) for op in operations) for t in (0, 1)]
    program_order = {edge for thread in threads for edge in zip(thread, thread[1:])}
    checked, failures, witness = 0, 0, None
    for schedule in interleave(*threads):
        lock, release_head = 0, None
        edges = set(program_order)
        valid = True
        for event in schedule:
            _, op = event
            acquire = op == "postload" or (op == "test" and protocol == "stock") or (op == "cas" and protocol == "acqcas")
            if op in ("test", "cas") and lock != 0:
                valid = False
                break
            if acquire and release_head is not None:
                edges.add((release_head, event))
            if op == "cas":
                lock = 1
                # RMW extends the preceding release sequence, even relaxed.
            elif op == "release":
                lock, release_head = 0, event
        if not valid:
            continue
        checked += 1
        a, b = (0, "payload"), (1, "payload")
        if not ordered(edges, a, b) and not ordered(edges, b, a):
            failures += 1
            witness = witness or schedule
    assert checked, "vacuous model: no successful claims"
    print(f"{protocol}: {failures}/{checked} schedules lack payload ordering")
    if witness:
        print("witness:", " -> ".join(f"T{t}.{op}" for t, op in witness))
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("stock", "acqcas", "postload", "all"), default="all")
    args = parser.parse_args()
    if args.protocol != "all":
        return 0 if check(args.protocol) else 1
    old = check("stock")
    cas = check("acqcas")
    post = check("postload")
    return 0 if not old and cas and post else 1


if __name__ == "__main__":
    raise SystemExit(main())
