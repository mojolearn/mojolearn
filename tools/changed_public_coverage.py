#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static test coverage of the endpoints `tools/changed_public_methods.py --endpoints` lists.

  python3 tools/changed_public_methods.py --endpoints fc8d425fb > endpoints.tsv
  python3 tools/changed_public_coverage.py endpoints.tsv > coverage.tsv
  COV_EXCLUDE=test_a.py,test_b.py python3 tools/changed_public_coverage.py endpoints.tsv   # leave files out

Per endpoint it counts test functions in python/mojolearn/tests/test_*.py (plus the qualification
modules a test drives case by case, _verify_graph.py and friends) that call or reference the method
with its class in scope: `real` without a fake binding or monkeypatch, `fake` with one, and `fault`
when the same function expects a refusal (pytest.raises, _raises, _refuses, native_failure, an
audited `.exception`). A heuristic: read every gap it reports by hand
(docs/plans/TEST_COVERAGE_20261021.md). Static, Python only: nothing is imported or run.
"""
import ast, sys, re, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT/"python/mojolearn/tests"
FAKE = re.compile(r"fake|Fake|FAKE|sabot|poison|Stub|stub|_Dummy|monkeypatch|SimpleNamespace|setattr")
ANCH = {"Program": {"capture", "load_program"}, "GenerationStream": {"stream", "stream_text", "resume_generation"},
        "CausalLMState": {"state"}, "TensorInfo": {"TensorInfo", "tensors", "info"}, "HFConfig": {"config", "from_pretrained"},
        "ModelPlan": {"plan_for"}, "Generator": {"Generator"}}
class TF:
    pass
funcs = []
VERIFY = [ROOT/"python/mojolearn"/f for f in ("_verify_graph.py", "_verify_generation.py",
          "_verify_model.py", "_verify_partitioned_program.py", "_verify_distributed_program.py")]
import os
EXCLUDE = set(filter(None, os.environ.get("COV_EXCLUDE", "").split(",")))
for path in [p for p in sorted(TESTS.glob("test_*.py")) if p.name not in EXCLUDE] + VERIFY:
    is_verify = path.name.startswith("_verify")
    src = path.read_text()
    try: tree = ast.parse(src)
    except SyntaxError: continue
    modnames = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): modnames.add(n.id)
        elif isinstance(n, ast.Attribute): modnames.add(n.attr)
        elif isinstance(n, ast.alias): modnames.add((n.asname or n.name).split(".")[-1]); modnames.add(n.name.split(".")[-1])
    # helpers: module-level non-test functions; propagate calls one level
    uses_getattr = "getattr(" in src
    module_strs = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for c in ast.walk(n.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str): module_strs.add(c.value)
    helpers = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef,)) and not n.name.startswith("test")}
    def facts(node, depth=0):
        calls, names, raises, fake = set(), set(), False, False
        seg = ast.get_source_segment(src, node) or ""
        if FAKE.search(seg) or any(a.arg == "monkeypatch" for a in getattr(node, "args", ast.arguments(args=[])).args): fake = True
        if re.search(r"pytest\.raises|_raises\(|assertRaises|_refuses|_refused|\.exception\b|native_failure|_failure\(|expect_refusal|_refusal\(", seg): raises = True
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute): calls.add(n.func.attr)
                elif isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
                    pass
            if isinstance(n, (ast.With, ast.AsyncWith)):
                calls.update(("__enter__", "__exit__"))
            if isinstance(n, ast.Subscript): calls.add("__getitem__")
            if isinstance(n, ast.Name):
                names.add(n.id)
                if depth < 3 and n.id in helpers and helpers[n.id] is not node and not isinstance(getattr(n, "ctx", None), ast.Store):
                    c, nm, r, f = facts(helpers[n.id], depth+1)
                    calls |= c; names |= nm; raises |= r; fake |= f
            elif isinstance(n, ast.Attribute): names.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n.value): names.add(n.value)
        return calls, names, raises, fake
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (node.name.startswith("test") or (is_verify and node in tree.body)):
            t = TF(); t.file = path.name; t.name = node.name
            t.calls, t.names, t.raises, t.fake = facts(node)
            if is_verify: t.fake = False
            t.modnames = modnames
            if uses_getattr:
                t.calls |= {x for x in t.names if isinstance(x, str)} | {x for x in module_strs if re.fullmatch(r"[a-z_][a-z0-9_]*", x)}
            # decorators: pytest.mark.parametrize string lists
            for d in node.decorator_list:
                for c in ast.walk(d):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        for w in re.split(r"[ ,]+", c.value):
                            t.calls.add(w)
            funcs.append(t)
rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))[1:]
out = csv.writer(sys.stdout, delimiter="\t")
out.writerow(["endpoint","status","real","fake","fault","files","real_files","fault_files"])
for kind, status, ep, line, helpers in rows:
    parts = ep.split(".")
    # find class component: first capitalized part after module
    cls = None; meth = parts[-1]
    for p in parts[1:-1]:
        if p[:1].isupper(): cls = p
    if meth == "__init__": meth = cls; anchor = None
    anchor = cls or parts[-2]
    anchors = {cls} | ANCH.get(cls, set()) if cls else set()
    real, fake, fault, files = [], [], [], set()
    for t in funcs:
        if meth not in t.calls and meth not in t.names: continue
        if cls and not any(a in t.names or a in t.modnames for a in anchors): continue
        if not cls and anchor not in t.modnames and meth not in t.names: continue
        files.add(t.file)
        (fake if t.fake else real).append(t)
        if t.raises: fault.append(t)
    rf = sorted({t.file for t in real}); ff = sorted({t.file for t in fault})
    out.writerow([ep, status, len(real), len(fake), len(fault), ",".join(sorted(files))[:300], ",".join(rf), ",".join(ff)])
