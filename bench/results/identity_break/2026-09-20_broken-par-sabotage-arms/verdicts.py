"""admit() and negative_control_moves() for each column, per part.

`admit()` is python/mojolearn/_verify_reference.py's; `negative_control_moves`
and `stable_digest` are tools/verification_matrix.py's. Nothing here weakens
`stable_digest`: it refuses a part with fewer than two repeats, which is why
every column below was run at --repeats 2.
"""
import os, sys, json
ROOT = "/Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/agent-a42cd6627cd863bdb"
sys.path.insert(0, os.path.join(ROOT, "tools"))
import verification_matrix as matrix
# load the reference module BY PATH, the way verification_evidence.py does:
# importing the package would need a built identical binary set.
ref = matrix.load("python/mojolearn/_verify_reference.py", "_arms_reference")

PARTS = ("train", "infer", "model", "batch")
LANES = ("par-ivf", "par-queries-nn", "par-rbf-sampler", "par-forecast-arima", "par-arima", "par-holtwinters", "par-queries-knn", "par-queries-radius")


def selfcheck():
    """stable_digest must refuse one repeat and accept two identical ones."""
    one = dict(verdict="STABLE", hashes=["a" * 16])
    two = dict(verdict="STABLE", hashes=["a" * 16, "a" * 16])
    moved = dict(verdict="STABLE", hashes=["a" * 16, "b" * 16])
    assert matrix.stable_digest(one, "train") is None, "one repeat was accepted"
    assert matrix.stable_digest(two, "train") == "a" * 16
    assert matrix.stable_digest(moved, "train") is None
    assert matrix.negative_control_moves(dict(verdict="STABLE", hashes=["b" * 16] * 2), two, "train")
    assert not matrix.negative_control_moves(two, two, "train")
    assert not matrix.negative_control_moves(dict(verdict="REFUSED", hashes=[], error="x"), two, "train"), \
        "a REFUSED cell was credited as a catch"
    assert matrix.negative_control_moves(dict(verdict="DIVERGENT", hashes=["c" * 16] * 2), two, "train"), \
        "a DIVERGENT cell carrying moved hashes was NOT credited"
    print("selfcheck: stable_digest refuses one repeat; a REFUSED cell is not a catch "
          "and a DIVERGENT one with moved hashes is\n")


def main():
    selfcheck()
    clean_path, sab_paths = sys.argv[1], sys.argv[2:]
    clean = json.load(open(clean_path))
    print("admit() verdicts")
    for p in [clean_path] + sab_paths:
        j = json.load(open(p))
        why = ref.admit(j, os.path.relpath(p, ROOT) if p.startswith(ROOT) else p)
        print("  %-52s %s" % (os.path.basename(p), why or "ADMITTED (None)"))
    print()
    for p in sab_paths:
        j = json.load(open(p))
        print("negative_control_moves, %s" % os.path.basename(p))
        for lane in LANES:
            key = lane + "/base"
            cell, base = j["cells"].get(key), clean["cells"].get(key)
            if cell is None or base is None:
                continue
            got = [part for part in PARTS if matrix.negative_control_moves(cell, base, part)]
            print("    %-20s verdict=%-10s credited parts: %s"
                  % (lane, cell.get("verdict"), ", ".join(got) or "NONE"))
        print()


main()
