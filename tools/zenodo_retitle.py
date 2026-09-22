#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Retitle every published Zenodo version under the mojolearn concept DOI.

The archived records carry the 2026-08 framing, which said Apple, because
that is what the library was when 0.1.0 was minted. It is now bitwise
identical across Apple Metal, NVIDIA CUDA and AMD HIP, and b031f2cd fixed
.zenodo.json and CITATION.cff so every FUTURE release is named correctly.
This handles the six that already exist.

A DOI IS PERMANENT AND PUBLIC, so this refuses to be casual:

  * dry run is the default; --apply is the only thing that writes.
  * every record is shown as current -> proposed before anything happens.
  * only `title` is touched. Not the creators, not the DOI, not the files,
    not the version, not the description. A retitle that quietly rewrote
    the author list would be much harder to notice than to cause.
  * a record already carrying the target title is skipped, so re-running
    after a partial failure costs nothing and repeats nothing.

Zenodo publishes metadata edits through a three-step flow: open the record
for editing, PUT the new metadata, publish. A record left open by a failure
half way through is visible in the account's uploads as a draft, so the
failure is loud rather than silent.

  python3 tools/zenodo_retitle.py                 # show what would change
  python3 tools/zenodo_retitle.py --apply         # do it
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CONCEPT_ID = "22068632"
TOKEN_FILE = os.path.expanduser("~/.mojolearn_zenodo_token")
API = "https://zenodo.org/api"


def target_title(repo_root):
    with open(os.path.join(repo_root, ".zenodo.json")) as fh:
        return json.load(fh)["title"]


def call(method, url, token, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def versions(token):
    """Every published version under the concept record, newest first."""
    url = "%s/records?q=conceptrecid:%s&all_versions=true&size=50&sort=newest" % (API, CONCEPT_ID)
    hits = call("GET", url, token).get("hits", {}).get("hits", [])
    return [
        {
            "id": h["id"],
            "doi": h.get("doi", ""),
            "version": (h.get("metadata") or {}).get("version", ""),
            "title": (h.get("metadata") or {}).get("title", ""),
        }
        for h in hits
    ]


def retitle(rec, token, want):
    """edit -> PUT title -> publish. Only `title` is replaced."""
    dep = "%s/deposit/depositions/%s" % (API, rec["id"])
    call("POST", dep + "/actions/edit", token)
    meta = call("GET", dep, token)["metadata"]
    meta["title"] = want
    call("PUT", dep, token, {"metadata": meta})
    call("POST", dep + "/actions/publish", token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the new titles (default is a dry run)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    want = target_title(repo_root)

    if not os.path.exists(TOKEN_FILE):
        sys.exit("no token at %s\n"
                 "Create one at https://zenodo.org/account/settings/applications/tokens/new/\n"
                 "with the deposit:actions and deposit:write scopes, then:\n"
                 "  umask 077; printf '%%s' '<token>' > %s" % (TOKEN_FILE, TOKEN_FILE))
    mode = os.stat(TOKEN_FILE).st_mode & 0o777
    if mode & 0o077:
        sys.exit("%s is mode %o; it holds a credential. chmod 600 it first." % (TOKEN_FILE, mode))
    with open(TOKEN_FILE) as fh:
        token = fh.read().strip()

    try:
        recs = versions(token)
    except urllib.error.URLError as exc:
        sys.exit("Zenodo did not answer: %s\n"
                 "If neutral hosts DO answer, the outage is theirs; try again later." % exc)

    if not recs:
        sys.exit("no versions came back for concept record %s. Refusing to guess." % CONCEPT_ID)

    print("target title, from .zenodo.json:")
    print("  %s\n" % want)
    todo = []
    for r in recs:
        same = r["title"] == want
        print("  %s  %s  version=%s" % (r["doi"], "OK  " if same else "EDIT", r["version"] or "?"))
        print("      now: %s" % r["title"])
        if not same:
            todo.append(r)
    print("\n%d record(s) already correct, %d to change." % (len(recs) - len(todo), len(todo)))

    if not todo:
        return
    if not args.apply:
        print("\nDRY RUN. Nothing was written. Re-run with --apply.")
        return

    for r in todo:
        print("  editing %s ..." % r["doi"], end=" ", flush=True)
        try:
            retitle(r, token, want)
        except urllib.error.HTTPError as exc:
            print("FAILED HTTP %s\n    %s" % (exc.code, exc.read().decode()[:400]))
            sys.exit("stopped at %s. Records before it are published; this one may be\n"
                     "left open as a draft in your Zenodo uploads. Check before re-running." % r["doi"])
        print("published")
    print("\n%d record(s) retitled." % len(todo))


if __name__ == "__main__":
    main()
