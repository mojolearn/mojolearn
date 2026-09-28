"""Explicit device-free dispatch bindings; never a general missing-image waiver.

bindings/_mojolearn_x_trees.mojo registers xtrees/api.mojo ensemble glue.
Tree fits are delegated by the Python API to _mojolearn_rf/_mojolearn_gbdt.
Both delegates must carry the requested image in the same numerical tier.
"""
import re

NO_IMAGE = "NONE-DELEGATED"
DEVICE_GLUE = {
    (tier, "_mojolearn_x_trees"): ("_mojolearn_rf", "_mojolearn_gbdt")
    for tier in ("fast", "identical")
}
ARCH = re.compile(rb"\b(?:sm_[0-9]+[a-z]*|compute_[0-9]+[a-z]*|gfx[0-9a-f]+)\b")


def validate(rows, root=None):
    """Reject unregistered no-image claims and absent/mismatched delegates."""
    indexed = {(r[0], r[1]): r[2] for r in rows if len(r) == 3}
    for row in rows:
        if len(row) != 3:
            raise ValueError("malformed architecture readback")
        tier, name, arch = row
        if arch != NO_IMAGE:
            continue
        delegates = DEVICE_GLUE.get((tier, name))
        if not delegates:
            raise ValueError(f"unregistered device-free binding: {tier}/{name}")
        images = [indexed.get((tier, delegate)) for delegate in delegates]
        if len(set(images)) != 1 or not images[0] or not re.fullmatch(r"sm_[0-9]+[a-z]*|gfx[0-9a-f]+", images[0]):
            raise ValueError(f"missing or mismatched device delegates: {tier}/{name}")
        if root is not None:
            binary = root / ("" if tier == "fast" else tier) / (name + ".so")
            if ARCH.search(binary.read_bytes()):
                raise ValueError(f"device-free witness contradicts binary: {binary}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if sys.argv[1] == "classify":
        print(NO_IMAGE if tuple(sys.argv[2:4]) in DEVICE_GLUE else "NONE")
    else:
        validate([line.split() for line in Path(sys.argv[2]).read_text().splitlines()])
