# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE BYTE CHECK FOR A LEG'S GENERATED-SOURCE COPIES (lane xcc-retire-1, 2026-11-27).

tools/generated_sources.py prepare keeps each invocation's generator inputs and outputs under
build/generated-sources/<run>/ (generated-inputs/<rel> and generated/<rel>). Those paths exist only on the machine
that prepared them, so a source inventory cannot list them and still match another machine's checkout (0.9.0's
linux-pack FileNotFoundError). They are never skipped unread: every reader checks their bytes here.

- generated-inputs/<rel> must equal the tracked <rel> in the same source root;
- generated/<rel> must be exactly what the tracked generator writes from its tracked inputs
  (tools/generated_sources.py expected_sha: regenerated in process, or a local generated copy verified against the
  generator).

Readers: packaging/linux/pack_wheel.py (a leg inventory that lists them), tools/check_linux_release_qualification.py
native_inventory and tools/linux_surface_qualification.sh (the box-side inventory: verified, then not listed).
"""
import hashlib
from pathlib import Path, PurePosixPath

#: the per-invocation evidence directory of tools/generated_sources.py
PREFIX = "build/generated-sources/"
KINDS = ("generated-inputs", "generated")


class GeneratedCopyError(ValueError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leg_generated_digest(source_root, name):
    """The sha256 hex the generated-source copy NAME (a POSIX path relative to SOURCE_ROOT) must have, or None when
    NAME is not one (other evidence under the directory, such as its manifest, is not source). Raises
    GeneratedCopyError when the expected bytes cannot be established (an unknown generated file, a missing tracked
    input, no generated copy to verify)."""
    parts = PurePosixPath(name).parts
    if len(parts) < 5 or parts[0] != "build" or parts[1] != "generated-sources" or parts[3] not in KINDS:
        return None
    root = Path(source_root)
    rel = "/".join(parts[4:])
    if parts[3] == "generated-inputs":
        if not (root / rel).is_file():
            raise GeneratedCopyError(f"{name}: its tracked source {rel} is not in the source root")
        return _sha(root / rel)
    import generated_sources  # tools/, on sys.path
    if rel not in generated_sources.GENERATED:
        raise GeneratedCopyError(f"{name}: {rel} is not a generated source (tools/generated_sources.py GENERATED)")
    try:
        inputs = {i: (root / i).read_bytes() for i in generated_sources.GENERATED[rel]}
    except OSError as exc:
        raise GeneratedCopyError(f"{name}: a tracked generator input is unreadable ({exc})") from exc
    candidates = [root / rel] + sorted((root / "build" / "generated-sources").glob("*/generated/" + rel))
    try:
        return generated_sources.expected_sha(rel, inputs, candidates)
    except generated_sources.GeneratedSourceError as exc:
        raise GeneratedCopyError(f"{name}: {exc}") from exc


def verify_copy(source_root, rel, path=None):
    """True when REL is a generated-source copy whose bytes (at PATH, default SOURCE_ROOT/REL) are what
    leg_generated_digest requires; False when REL is not a copy; raises GeneratedCopyError on a differing byte."""
    want = leg_generated_digest(source_root, rel)
    if want is None:
        return False
    got = _sha(Path(path) if path else Path(source_root) / rel)
    if got != want:
        raise GeneratedCopyError(f"{rel}: sha256 {got[:12]} differs from its tracked source or generator ({want[:12]})")
    return True
