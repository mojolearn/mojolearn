#!/usr/bin/env python3
"""THE USER'S INSTALL, FROM THE INDEX (2026-09-26). Stdlib only: it runs on
the Mac and ships alone to a rented box.

The release columns install LOCAL wheel files. A user runs
`pip install mojolearn==V` against an index, and that is where the split
Linux release can break: the core requires mojolearn-nvidia==V and
mojolearn-amd==V, each plugin requires mojolearn==V back (a cycle pip must
resolve), and the three are uploaded in order by separate jobs. Two
subcommands:

  precheck --index testpypi|pypi --version V
      On the Mac, before anything is rented: all three projects serve V on
      the index (the JSON API), each with a manylinux x86_64 wheel, none
      yanked, and where the index reports Requires-Dist, the exact pins of
      the cycle. Refuses by project name.

  verify --index testpypi|pypi --version V --vendor cuda|hip --report R --out J
      On the box, run by the python of the venv pip installed into:
      mojolearn, mojolearn-nvidia and mojolearn-amd are installed at exactly
      V (importlib.metadata); in pip's --report R each of the three came from
      the index's own file host (TestPyPI: test-files.pythonhosted.org) and
      every other distribution from PyPI's (files.pythonhosted.org), so a
      same-named project on the other index cannot stand in for ours
      (dependency confusion); nothing was a direct URL; and `import
      mojolearn` loads the GPU set of this box's vendor from that vendor's
      plugin at V. Writes J with the verdict and every fact it judged.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

#: The three projects of the split Linux release (python/mojolearn/gpu_plugins.py,
#: spelled here because this file ships to the box without the package).
CORE = "mojolearn"
PLUGINS = {"cuda": "mojolearn-nvidia", "hip": "mojolearn-amd"}
PROJECTS = (CORE, PLUGINS["cuda"], PLUGINS["hip"])

#: index -> (simple index, JSON API root, file host of OUR three projects)
INDEXES = {
    "testpypi": ("https://test.pypi.org/simple/", "https://test.pypi.org/pypi", "test-files.pythonhosted.org"),
    "pypi": ("https://pypi.org/simple/", "https://pypi.org/pypi", "files.pythonhosted.org"),
}
#: where every other dependency (numpy, ...) must come from, on either index
PYPI_FILES = "files.pythonhosted.org"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_prefix(project):
    return project.replace("-", "_")


def requirement_pins(requires_dist):
    """{normalized name: exact version} of the `name==V` requirements that
    carry no marker and no extra."""
    pins = {}
    for req in requires_dist or []:
        m = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*\(?\s*==\s*([^\s;),]+)\s*\)?\s*", req)
        if m and not m.group(2):
            pins[norm(m.group(1))] = m.group(3)
    return pins


def fetch_json(url, timeout=20):
    """(HTTP status, parsed body or None)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:  # network, TLS, a body that is not JSON
        return 0, repr(exc)


def fetch_text(url, timeout=20):
    """The body as text, or None."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def precheck(index, version, api=None, fetch=fetch_json, fetch_meta=fetch_text):
    """(lines, problems) for the three projects at `version` on `index`."""
    api = (api or os.environ.get("MOJOLEARN_INDEX_JSON_API") or INDEXES[index][1]).rstrip("/")
    lines, problems = [], []
    for project in PROJECTS:
        url = f"{api}/{project}/{version}/json"
        code, doc = fetch(url)
        if code != 200 or not isinstance(doc, dict):
            why = f"HTTP {code}" if code else f"no answer ({doc})"
            lines.append(f"  {project}=={version}: MISSING on {index} ({why}, {url})")
            problems.append(f"{project}=={version} is not on {index} ({why})")
            continue
        files = doc.get("urls") or []
        wheels = [f.get("filename", "") for f in files
                  if f.get("packagetype") == "bdist_wheel"
                  and f.get("filename", "").startswith(f"{wheel_prefix(project)}-{version}-")]
        linux = [w for w in wheels if re.search(r"-manylinux[0-9a-z_]*_x86_64\.whl$", w)]
        yanked = [f.get("filename", "") for f in files if f.get("yanked")]
        info = doc.get("info") or {}
        requires = info.get("requires_dist")
        # The Linux wheel's own METADATA (PEP 658, served beside the file) is
        # what pip reads; prefer it to the project-level summary (the JSON API
        # does not say whether it is served, so it is simply asked for).
        entry = next((f for f in files if f.get("filename") in linux[:1]), None)
        if entry and entry.get("url"):
            text = fetch_meta(entry["url"] + ".metadata")
            if text is not None and text.lstrip().startswith("Metadata-Version"):
                requires = [line.split(":", 1)[1].strip() for line in text.splitlines()
                            if line.lower().startswith("requires-dist:")]
        info = dict(info, requires_dist=requires)
        pins = requirement_pins(requires)
        facts = f"{len(files)} file(s), manylinux x86_64: {', '.join(linux) or 'NONE'}"
        if not linux:
            problems.append(f"{project}=={version} on {index} has no manylinux x86_64 wheel (files: "
                            f"{', '.join(f.get('filename', '') for f in files) or 'none'})")
        if yanked:
            problems.append(f"{project}=={version} on {index} has yanked file(s): {', '.join(yanked)}")
        if info.get("requires_dist") is None:
            facts += "; Requires-Dist not reported by the index (the box's resolution checks the pins)"
        elif project == CORE:
            want = {norm(p): version for p in PLUGINS.values()}
            got = {k: v for k, v in pins.items() if k in want}
            if got != want:
                problems.append(f"{CORE}=={version} on {index} requires {got or 'no plugin'}, not both plugins "
                                f"at =={version} (this is not the split core: a combined wheel, or a broken pin)")
            facts += f"; requires {', '.join(f'{k}=={v}' for k, v in sorted(got.items())) or 'no plugin'}"
        else:
            if pins.get(CORE) != version:
                problems.append(f"{project}=={version} on {index} does not require {CORE}=={version} "
                                f"(requires {info.get('requires_dist')})")
            facts += f"; requires {CORE}=={pins.get(CORE, '?')}"
        lines.append(f"  {project}=={version}: on {index}, {facts}")
    return lines, problems


def _url_host(url):
    m = re.match(r"[a-z+]+://([^/:@]+@)?([^/:]+)", url or "")
    return m.group(2).lower() if m else ""


def judge_report(report, index, version):
    """(rows, problems) over a `pip install --report` document."""
    ours = INDEXES[index][2]
    rows, problems, seen = [], [], {}
    for item in report.get("install") or []:
        meta = item.get("metadata") or {}
        name, got = norm(meta.get("name", "")), meta.get("version", "")
        info = item.get("download_info") or {}
        url = info.get("url", "")
        host = _url_host(url)
        want = ours if name in PROJECTS else PYPI_FILES
        rows.append(dict(name=name, version=got, host=host, url=url, expected_host=want,
                         sha256=((info.get("archive_info") or {}).get("hashes") or {}).get("sha256")
                         or (info.get("archive_info") or {}).get("hash")))
        seen[name] = got
        if item.get("is_direct"):
            problems.append(f"{name} {got} was a direct URL requirement ({url}), not resolved from the index")
        if host != want:
            problems.append(f"{name} {got} came from {host or url or 'nowhere'}, not {want}"
                            + (" (DEPENDENCY CONFUSION GUARD: every non-mojolearn distribution must come "
                               "from PyPI)" if name not in PROJECTS else ""))
        if name.startswith("mojolearn") and name not in PROJECTS:
            problems.append(f"an unexpected mojolearn-named distribution {name} {got} was installed")
    for project in PROJECTS:
        if project not in seen:
            problems.append(f"{project} is not in pip's report: pip did not install it")
        elif seen[project] != version:
            problems.append(f"pip installed {project} {seen[project]}, not {version}")
    return rows, problems


def load_check(vendor, version):
    """(facts, problems): import mojolearn and read which set it loaded."""
    facts, problems = {}, []
    try:
        import mojolearn
        from mojolearn import _backend
        facts["version"] = mojolearn.__version__
        facts["package"] = os.path.dirname(os.path.realpath(mojolearn.__file__))
        facts["vendor"] = _backend.vendor()
        facts["vendor_how"] = str(getattr(_backend, "vendor_how", lambda: None)())
        facts["gpu_arch"] = getattr(_backend, "gpu_arch", lambda: None)()
        facts["plugin"] = getattr(_backend, "gpu_plugin", lambda: None)()
        mode = _backend.default_mode()
        facts["mode"] = mode
        facts["tier_dir"] = os.path.realpath(_backend.tier_dir(mode))
    except Exception as exc:
        problems.append(f"import mojolearn failed: {exc!r}")
        return facts, problems
    if facts["version"] != version:
        problems.append(f"import mojolearn gave version {facts['version']}, not {version}")
    if facts["vendor"] != vendor:
        problems.append(f"mojolearn loaded vendor {facts['vendor']}, not {vendor} (vendor_how: {facts['vendor_how']})")
    plugin = facts["plugin"] or {}
    if plugin.get("distribution") != PLUGINS[vendor] or plugin.get("version") != version:
        problems.append(f"the loaded GPU set is not from {PLUGINS[vendor]} {version} (gpu_plugin() = {facts['plugin']})")
    want_dir = os.path.join(facts["package"], vendor) + os.sep
    if not (facts["tier_dir"] + os.sep).startswith(want_dir):
        problems.append(f"the loaded tier directory {facts['tier_dir']} is not under {want_dir}")
    return facts, problems


def verify(index, version, vendor, report_path, out):
    from importlib import metadata
    doc = dict(index=index, version=version, vendor=vendor, installed={}, report=[], load={}, problems=[])
    for project in PROJECTS:
        try:
            doc["installed"][project] = metadata.version(project)
        except metadata.PackageNotFoundError:
            doc["installed"][project] = None
        if doc["installed"][project] != version:
            doc["problems"].append(f"{project} is installed at {doc['installed'][project]}, not {version}")
    try:
        report = json.load(open(report_path))
    except (OSError, ValueError) as exc:
        report = None
        doc["problems"].append(f"no pip report at {report_path} ({exc})")
    if report is not None:
        doc["pip_version"] = report.get("pip_version")
        doc["report"], problems = judge_report(report, index, version)
        doc["problems"] += problems
    doc["load"], problems = load_check(vendor, version)
    doc["problems"] += problems
    doc["verdict"] = "FAILED" if doc["problems"] else "PASSED"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    for project in PROJECTS:
        print(f"installed {project} {doc['installed'][project]}")
    for row in doc["report"]:
        print(f"report {row['name']} {row['version']} from {row['host']}")
    print(f"load vendor={doc['load'].get('vendor')} plugin={doc['load'].get('plugin')} tier_dir={doc['load'].get('tier_dir')}")
    for p in doc["problems"]:
        print("PROBLEM: " + p)
    print(f"verdict={doc['verdict']}")
    return 0 if doc["verdict"] == "PASSED" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precheck")
    p.add_argument("--index", choices=sorted(INDEXES), required=True)
    p.add_argument("--version", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--index", choices=sorted(INDEXES), required=True)
    v.add_argument("--version", required=True)
    v.add_argument("--vendor", choices=sorted(PLUGINS), required=True)
    v.add_argument("--report", required=True)
    v.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "precheck":
        lines, problems = precheck(args.index, args.version)
        print(f"index precheck: {', '.join(PROJECTS)} at {args.version} on {args.index}")
        print("\n".join(lines), flush=True)
        for problem in problems:
            print("REFUSED: " + problem, file=sys.stderr)
        if not problems:
            print(f"index precheck PASSED: all three projects serve {args.version} on {args.index}")
        return int(bool(problems))
    return verify(args.index, args.version, args.vendor, args.report, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
