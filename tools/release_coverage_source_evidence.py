"""The verifier coverage audit as ledger evidence for coverage-source.

Used by tools/release_ledger_admission.py. The actual producer is
tools/verify_coverage_audit.py --json (static analysis; nothing is built or
run). Row statuses are derived here by the tested revision's own checker,
extracted from Git into a temporary directory and run in an isolated
interpreter, so a later checkout's annotations cannot relabel an old audit.

Coverage ID "coverage mapping and exclusions reviewed" counts every audited
native registration and public method row. It passes only when every native
row is covered or explicitly not applicable, every native fault arm is armed
or explicitly not applicable, and every public method row is covered (directly
or via its native entries) or explicitly not applicable. GAP and owed rows
refuse admission. A ``changed`` audit must use the last published release's
source commit, recorded in the tested revision's release verification records,
as its base; a symbolic base such as origin/main is refused. A whole-surface
(``all``) audit needs no base.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from release_ledger_admission import require, reference, checked, git

TOOL = "tools/release_coverage_source_evidence.py"
COVERAGE = "coverage mapping and exclusions reviewed"
NATIVE_PASS = {"covered", "n/a"}
SABOTAGE_PASS = {"armed", "n/a"}
PYTHON_PASS = {"covered", "covered (via native)", "n/a"}
CHECKER = ("tools/verify_coverage_audit.py", "tools/mojo_registrations.py", "tools/verify_reference_evidence.py")
HEX40 = re.compile(r"[0-9a-f]{40}")
DRIVER = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import verify_coverage_audit as V
audit = json.load(open(sys.argv[2]))
native = audit["native"]
by_entry = V.native_status_by_entry(native)
print(json.dumps(dict(
    native=[[r["entry"], r.get("src"), V.triage(r)[0], V.sabotage_of(r)[0]] for r in native],
    python=[[r["public"], r["op"], V.py_triage(r, by_entry)[0]] for r in audit["python"]])))
"""


def supports_requirement(wanted):
    return (wanted.get("kind") == "source" and wanted.get("coverage") == [COVERAGE]
            and wanted.get("inputs", {}).get("contract", {}).get("id") == "coverage-source"
            and not wanted.get("missing_inputs"))


def _version(text):
    return tuple(int(part) for part in text.split("."))


def published_base(root, revision, version):
    """Source commit of the newest published release before VERSION, at REVISION."""
    listing = git(root, "ls-tree", "-r", "--name-only", revision, "bench/results/release_verification/")
    best = None
    for path in listing.splitlines():
        if not path.endswith("/release-provenance.json"):
            continue
        record = json.loads(git(root, "show", revision + ":" + path))
        published = str(record.get("version", ""))
        if (record.get("schema") == "mojolearn.release-provenance.v1" and re.fullmatch(r"\d+(\.\d+)+", published)
                and _version(published) < _version(version) and HEX40.fullmatch(str(record.get("source_commit")))):
            if best is None or _version(published) > _version(best[0]):
                best = (published, record["source_commit"])
    require(best is not None, "no published release record precedes this version at the tested revision")
    return best


def statuses(root, revision, audit_path):
    """Row statuses from the tested revision's own checker files."""
    with tempfile.TemporaryDirectory(prefix="mojolearn-coverage-checker-") as temporary:
        base = Path(temporary)
        names = list(CHECKER) + [p for p in git(root, "ls-tree", "-r", "--name-only", revision,
                                                  "tools/identity_lanes/").splitlines() if p]
        for name in names:
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "-C", str(root), "show", revision + ":" + name]))
        process = subprocess.run([sys.executable, "-I", "-c", DRIVER, str(base / "tools"), str(audit_path)],
                                 cwd=temporary, capture_output=True, text=True)
        diagnostic = (process.stderr.strip().splitlines() or ["no diagnostic"])[-1]
        require(process.returncode == 0, "tested checker could not classify the audit: " + diagnostic)
        return json.loads(process.stdout.strip().splitlines()[-1])


def _facts(root, producer, version):
    audit_ref = reference(producer["audit"])
    audit = checked(audit_ref)
    head = audit.get("head")
    require(isinstance(head, str) and HEX40.fullmatch(head), "audit lacks its exact source revision")
    require(isinstance(audit.get("native"), list) and isinstance(audit.get("python"), list),
            "audit lacks native/public method rows")
    scope = audit.get("scope")
    base = published = None
    if scope == "changed":
        published = published_base(root, head, version)
        named = str(audit.get("base") or "")
        require(named and "/" not in named, "changed-scope audit base must be a fixed commit, not " + repr(named))
        try:
            base = git(root, "rev-parse", "--verify", "--quiet", named + "^{commit}").strip()
        except subprocess.CalledProcessError as error:
            raise ValueError("changed-scope audit base does not resolve: " + named) from error
        require(base == published[1], "changed-scope audit base is not the last published release source commit")
    else:
        require(scope == "all", "unknown audit scope")
    rows = statuses(root, head, Path(audit_ref["path"]))
    require(len(rows["native"]) == len(audit["native"]) and len(rows["python"]) == len(audit["python"]),
            "checker classified another row set")
    failing = ([dict(kind="native", entry=e, src=s, status=t) for e, s, t, _ in rows["native"] if t not in NATIVE_PASS]
               + [dict(kind="sabotage", entry=e, src=s, status=a) for e, s, _, a in rows["native"] if a not in SABOTAGE_PASS]
               + [dict(kind="python", public=p, op=o, status=t) for p, o, t in rows["python"] if t not in PYTHON_PASS])
    def counts(values):
        result = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))
    return dict(producer="coverage-source", source_root=str(Path(root).resolve()), source_revision=head,
                audit=audit_ref, scope=scope, base=base, published=list(published) if published else None,
                version=version, rows=len(rows["native"]) + len(rows["python"]),
                native=counts(t for _, _, t, _ in rows["native"]),
                sabotage=counts(a for _, _, _, a in rows["native"]),
                python=counts(t for _, _, t in rows["python"]),
                failing=len(failing), failing_examples=failing[:50])


def collect(root, document, producer):
    facts = _facts(root, producer, document["version"])
    modes = document.get("configuration", {}).get("release_modes")
    require(modes == ["identical"], "only the identical release is admitted")
    return dict(source_revision=facts["source_revision"], facts=facts, units={"coverage-source": {}},
                spec_ids=["coverage-source"], declared=["release_modes"], configuration=dict(release_modes=modes))


def check_unit(requirement_id, wanted, facts, unit):
    require(requirement_id == "coverage-source" and supports_requirement(wanted), "another source contract")
    require(facts["failing"] == 0, f"{facts['failing']} audited rows are GAP/owed/unreviewed; first: "
            + json.dumps(facts["failing_examples"][:3]))
    require(facts["rows"] > 0, "audit has no rows to review")
    return ([dict(id=COVERAGE, status="passed", exit_code=0, expected=facts["rows"], completed=facts["rows"],
                  failed=0, skipped=0)], [facts["audit"]])


class _Adapter:
    collect = staticmethod(collect)
    check_unit = staticmethod(check_unit)

    @staticmethod
    def verify(report):
        return _facts(report["facts"]["source_root"], report["producer"], report["facts"]["version"])


ADAPTERS = {"coverage-source": _Adapter()}
