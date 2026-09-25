#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The release PASS LEDGER in R2 (2026-09-25): every build leg and every
column result, recorded once, keyed by WHAT WAS TESTED rather than by the
commit that asked.

    python3 tools/release_ledger.py list [prefix]      what the ledger holds
    python3 tools/release_ledger.py get <key>

Keys (under release-ledger/v1/ in the mojolearn-data bucket):
  build/<vendor>-<arch>/<set identity digest>/<tooling digest>.json
      a Linux build leg: the set's binding identities (release_reuse
      set_identity) and the build tooling that ran (release_tooling)
  column/<vendor>/<wheel sha256>/<lane selection digest>.json
      an NVIDIA or AMD wheel column: the exact wheel bytes, the vendor and the
      lanes it ran
  column/metal/<source commit>.json
      the Apple column of a source commit (it runs from the source, not a wheel)
Each value records the verdict, the evidence paths and digests, and both
commits (source_commit, tooling_commit). An entry is written once: a second
PASS for the same key is not rewritten.

The ledger is an index, never evidence: tools/release.py admits a ledger PASS
only when the evidence it names is on this machine and its digests verify, and
otherwise runs the leg or column again. The credentials are ~/.mojolearn_r2
(MOJOLEARN_R2_CREDS moves it; docs/REMOTE_DATA_R2.md); without them, or with
MOJOLEARN_RELEASE_LEDGER=off, the ledger is the local mirror alone
(<evidence>/release/ledger/), which tools/release.py also keeps as a cache.
"""
import argparse
import json
import os
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "release-ledger/v1/"
SCHEMA = "mojolearn.release-ledger-entry.v1"


def build_key(vendor, arch, set_digest, tooling_digest):
    return f"build/{vendor}-{arch}/{set_digest}/{tooling_digest}.json"


def column_key(vendor, wheel_sha256, selection_digest):
    return f"column/{vendor}/{wheel_sha256}/{selection_digest}.json"


def apple_key(source_commit):
    return f"column/metal/{source_commit}.json"


def read_creds(path):
    """KEY=VALUE lines of the shell file ~/.mojolearn_r2 (export and quotes allowed)."""
    creds = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        creds[k.strip()] = v.strip().strip("'\"")
    need = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if not all(creds.get(k) for k in need):
        return None
    return {k: creds[k] for k in need}


class R2:
    """The bucket through tools/bincache.py's SigV4 client."""

    def __init__(self, creds):
        sys.path.insert(0, str(ROOT / "tools"))
        import bincache
        self.b, self.creds = bincache, creds

    def get(self, key):
        try:
            _, body = self.b.r2_request(self.creds, "GET", key)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def put(self, key, body):
        self.b.r2_request(self.creds, "PUT", key, headers={"content-type": "application/json"}, data=body)

    def list(self, prefix):
        return [k for k, _, _ in self.b.list_objects(self.creds, prefix)]


class Ledger:
    """Reads and writes through R2 when there is a client, and always through
    the local mirror. `client` is anything with get/put/list (tests pass a
    dict-backed stand-in)."""

    def __init__(self, local_dir, client=None, say=None):
        self.local = Path(local_dir)
        self.client = client
        self.say = say or (lambda m: None)
        self.errors = []
        #: False keeps a read from writing the mirror (release.py --status)
        self.cache = True

    @classmethod
    def open(cls, evidence_root, say=None):
        local = Path(evidence_root) / "release" / "ledger"
        if os.environ.get("MOJOLEARN_RELEASE_LEDGER", "").lower() in ("off", "0", "local"):
            return cls(local, None, say)
        path = os.path.expanduser(os.environ.get("MOJOLEARN_R2_CREDS", "~/.mojolearn_r2"))
        try:
            creds = read_creds(path)
        except OSError:
            creds = None
        return cls(local, R2(creds) if creds else None, say)

    @property
    def where(self):
        return ("R2 " + PREFIX if self.client else "local only") + f" (mirror {self.local})"

    def _remote(self, what, fn, *a):
        if not self.client:
            return None
        try:
            return fn(*a)
        except Exception as exc:  # a ledger outage never stops a release; it only costs a rerun
            self.errors.append(f"{what}: {type(exc).__name__}: {exc}")
            self.say(f"  ledger: {what} failed ({type(exc).__name__}); using the local mirror")
            return None

    def get(self, key):
        p = self.local / key
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            pass
        body = self._remote("get " + key, self.client.get if self.client else None, PREFIX + key)
        if body is None:
            return None
        try:
            doc = json.loads(body)
        except ValueError:
            return None
        if self.cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return doc

    def put(self, key, entry):
        """Record ENTRY once. An existing PASS under KEY is kept, not rewritten."""
        old = self.get(key)
        if old and old.get("verdict") == "PASS":
            return old
        doc = dict(entry, schema=SCHEMA, key=key)
        body = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        p = self.local / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        self._remote("put " + key, self.client.put if self.client else None, PREFIX + key, body.encode())
        return doc

    def find(self, prefix):
        """Every entry under PREFIX (a key prefix), R2 and the mirror together."""
        keys = set()
        base = self.local / prefix
        root = base if base.is_dir() else base.parent
        if root.is_dir():
            keys |= {p.relative_to(self.local).as_posix() for p in root.rglob("*.json")
                     if p.relative_to(self.local).as_posix().startswith(prefix)}
        listed = self._remote("list " + prefix, self.client.list if self.client else None, PREFIX + prefix)
        keys |= {k[len(PREFIX):] for k in (listed or []) if k.startswith(PREFIX)}
        out = []
        for k in sorted(keys):
            doc = self.get(k)
            if doc:
                out.append(doc)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("list", "get"))
    ap.add_argument("arg", nargs="?", default="")
    a = ap.parse_args(argv)
    ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
    led = Ledger.open(ev, say=print)
    print(f"# ledger: {led.where}")
    if a.cmd == "get":
        print(json.dumps(led.get(a.arg), indent=1, sort_keys=True))
        return 0
    for doc in led.find(a.arg):
        print(f"{doc.get('verdict', '?'):5} {doc.get('key')}  source {str(doc.get('source_commit'))[:12]} "
              f"tooling {str(doc.get('tooling_commit'))[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
