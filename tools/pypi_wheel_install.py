#!/usr/bin/env python3
"""Install our released wheels from PyPI with byte provenance, and prove they loaded.

    python3 tools/pypi_wheel_install.py install --version 0.8.44 --vendor nvidia \
        --python <venv>/bin/python --dir <cache>/pypi-wheels --manifest <out>/wheel-manifest.json
    python3 tools/pypi_wheel_install.py verify --manifest <out>/wheel-manifest.json --python <venv>/bin/python

install: `pip download mojolearn==V mojolearn-<vendor>==V --no-deps` from https://pypi.org/simple into
an empty directory, hash every file (SHA-256), check each hash against PyPI's JSON API digest
(https://pypi.org/pypi/<name>/<V>/json), then `pip install` those exact files (a dependency pass, then a
`--force-reinstall --no-deps` pass so an older install of the same version cannot stand in for them).
Writes the manifest: requested specs, index URL, file names, sha256, the PyPI digest and whether it matched.

verify: refuses (exit 2, reason on stderr) unless, in the given interpreter, mojolearn and the vendor
plugin import from site-packages, their installed versions equal V, and pip's PEP 610 direct_url.json
for each names the manifest's file (and its hash when pip recorded one). bench_board.py runs the same
check before any timing.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

INDEX_URL = "https://pypi.org/simple"
JSON_API = "https://pypi.org/pypi/%s/%s/json"
SCHEMA = "mojolearn-pypi-wheel-manifest/1"
PLUGIN = {"nvidia": "mojolearn-nvidia", "amd": "mojolearn-amd"}


def distributions(vendor):
    if vendor not in PLUGIN:
        raise ValueError("vendor must be nvidia or amd, not %r" % vendor)
    return ["mojolearn", PLUGIN[vendor]]


def wheel_prefix(name, version):
    """A wheel file name starts with the normalized distribution name and the version."""
    return "%s-%s-" % (name.replace("-", "_").lower(), version)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def pypi_digests(name, version, fetch=fetch_json):
    """{filename: sha256} for one release, from PyPI's JSON API."""
    doc = fetch(JSON_API % (name, version))
    return {u["filename"]: (u.get("digests") or {}).get("sha256") for u in doc.get("urls") or []}


def run(cmd, log):
    with open(log, "ab") as fh:
        fh.write(("\n$ %s\n" % " ".join(cmd)).encode())
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def pick_file(directory, name, version):
    pre = wheel_prefix(name, version)
    got = sorted(f for f in os.listdir(directory) if f.lower().startswith(pre) and f.endswith(".whl"))
    if len(got) != 1:
        raise SystemExit("pypi_wheel_install: expected one %s==%s wheel in %s, found %s"
                         % (name, version, directory, got or "none"))
    return os.path.join(directory, got[0])


def build_manifest(version, vendor, directory, fetch=fetch_json, index_url=INDEX_URL):
    """The manifest for the files already downloaded into `directory`."""
    files = []
    for name in distributions(vendor):
        path = pick_file(directory, name, version)
        fname = os.path.basename(path)
        digest = sha256_file(path)
        published = pypi_digests(name, version, fetch).get(fname)
        files.append(dict(distribution=name, requested="%s==%s" % (name, version), file=fname,
                          path=os.path.abspath(path), bytes=os.path.getsize(path), sha256=digest,
                          pypi_sha256=published, pypi_digest_match=bool(published) and published == digest))
    return dict(schema=SCHEMA, version=version, vendor=vendor, index_url=index_url,
                json_api=JSON_API % ("<name>", version), requested=[f["requested"] for f in files],
                files=files, all_digests_match=all(f["pypi_digest_match"] for f in files),
                created=time.time())


def install(args):
    os.makedirs(args.dir, exist_ok=True)
    stale = [f for f in os.listdir(args.dir) if f.endswith(".whl")]
    if stale:
        raise SystemExit("pypi_wheel_install: %s already holds wheels (%s); use an empty directory"
                         % (args.dir, ", ".join(stale)))
    log = args.log or os.path.join(args.dir, "pypi-wheel-install.log")
    specs = ["%s==%s" % (n, args.version) for n in distributions(args.vendor)]
    pip = [args.python, "-m", "pip"]
    common = ["--no-input", "--disable-pip-version-check"]
    if run(pip + ["download"] + common + ["--no-deps", "--index-url", INDEX_URL, "-d", args.dir] + specs, log):
        raise SystemExit("pypi_wheel_install: pip download failed; see " + log)
    manifest = build_manifest(args.version, args.vendor, args.dir)
    bad = [f["file"] for f in manifest["files"] if not f["pypi_digest_match"]]
    if bad:
        _write(args.manifest, manifest)
        raise SystemExit("pypi_wheel_install: SHA-256 differs from PyPI's digest for " + ", ".join(bad))
    paths = [f["path"] for f in manifest["files"]]
    # Pass 1 resolves the dependencies; pass 2 puts exactly these bytes in place even when the
    # same version was already installed from somewhere else.
    if run(pip + ["install"] + common + paths, log):
        raise SystemExit("pypi_wheel_install: pip install failed; see " + log)
    if run(pip + ["install"] + common + ["--force-reinstall", "--no-deps"] + paths, log):
        raise SystemExit("pypi_wheel_install: pip reinstall of the verified files failed; see " + log)
    manifest["installed_into"] = args.python
    manifest["loaded"] = loaded_receipt(args.python, manifest)
    _write(args.manifest, manifest)
    why = refusal(manifest, manifest["loaded"])
    if why:
        raise SystemExit("pypi_wheel_install: " + why)
    print("pypi_wheel_install: installed %s (sha256 match PyPI)" % ", ".join(f["file"] for f in manifest["files"]))
    return 0


_LOADED_PROBE = r"""
import importlib, importlib.metadata as md, json, os, sys
names = json.loads(sys.argv[1])
out = {}
for name in names:
    rec = {}
    try:
        dist = md.distribution(name)
        rec["version"] = dist.version
        rec["location"] = str(dist.locate_file(""))
        txt = dist.read_text("direct_url.json")
        rec["direct_url"] = json.loads(txt) if txt else None
    except Exception as e:
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    out[name] = rec
try:
    import mojolearn
    out["mojolearn"]["module_file"] = os.path.realpath(mojolearn.__file__)
except Exception as e:
    out["mojolearn"]["import_error"] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
"""


def loaded_receipt(python, manifest):
    names = [f["distribution"] for f in manifest["files"]]
    p = subprocess.run([python, "-c", _LOADED_PROBE, json.dumps(names)], capture_output=True, text=True,
                       timeout=300)
    if p.returncode:
        return {"error": (p.stderr or p.stdout or "rc %d" % p.returncode)[-1500:]}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": "unreadable probe output: " + p.stdout[-500:]}


def _in_site_packages(path):
    return bool(path) and ("site-packages" in path or "dist-packages" in path)


def refusal(manifest, loaded):
    """None when every manifest file is what the interpreter loaded; otherwise the reason."""
    if not manifest or manifest.get("schema") != SCHEMA:
        return "no PyPI wheel manifest"
    if not manifest.get("all_digests_match"):
        return "the manifest's files do not match PyPI's digests"
    if not loaded or loaded.get("error"):
        return "could not read the installed distributions: %s" % (loaded or {}).get("error")
    version = manifest["version"]
    core = loaded.get("mojolearn") or {}
    if core.get("import_error"):
        return "mojolearn does not import: " + core["import_error"]
    if not _in_site_packages(core.get("module_file")):
        return "mojolearn loaded from %s, not site-packages" % core.get("module_file")
    for f in manifest["files"]:
        rec = loaded.get(f["distribution"]) or {}
        if rec.get("error"):
            return "%s is not installed: %s" % (f["distribution"], rec["error"])
        if rec.get("version") != version:
            return "%s installed version %s, requested %s" % (f["distribution"], rec.get("version"), version)
        if not _in_site_packages(rec.get("location")):
            return "%s is installed at %s, not site-packages" % (f["distribution"], rec.get("location"))
        du = rec.get("direct_url") or {}
        if os.path.basename(str(du.get("url") or "")) != f["file"]:
            return "%s was not installed from the verified file %s (direct_url %s)" % (
                f["distribution"], f["file"], du.get("url"))
        h = (du.get("archive_info") or {}).get("hashes", {}).get("sha256")
        if not h:
            legacy = str((du.get("archive_info") or {}).get("hash") or "")
            h = legacy.split("=", 1)[1] if legacy.startswith("sha256=") else None
        if h and h != f["sha256"]:
            return "%s installed bytes hash %s, verified file %s" % (f["distribution"], h, f["sha256"])
    return None


def load_manifest(path):
    with open(path) as fh:
        return json.load(fh)


def verify(manifest_path, python):
    """(manifest with a fresh `loaded` receipt, refusal or None)."""
    manifest = load_manifest(manifest_path)
    loaded = loaded_receipt(python, manifest)
    manifest = dict(manifest, loaded=loaded)
    return manifest, refusal(manifest, loaded)


def _write(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("install")
    a.add_argument("--version", required=True)
    a.add_argument("--vendor", choices=sorted(PLUGIN), required=True)
    a.add_argument("--python", default=sys.executable)
    a.add_argument("--dir", required=True, help="empty download directory")
    a.add_argument("--manifest", required=True)
    a.add_argument("--log", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    if args.cmd == "install":
        return install(args)
    _, why = verify(args.manifest, args.python)
    if why:
        print("pypi_wheel_install: REFUSING: " + why, file=sys.stderr)
        return 2
    print("pypi_wheel_install: verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
