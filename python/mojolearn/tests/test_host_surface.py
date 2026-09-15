# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU surface manifest (`mojolearn/host_surface.py`) against the tree.

One file declares the CPU surface; these tests are what keeps the tree from
drifting away from it (the host surface manifest lane, 2026-09-14). Each one
reads SOURCE, never a built binary, so it runs on a box with nothing built:

  * every family the manifest lists has a binding source under bindings/,
    and every `bindings/_mojolearn_*_host.mojo` is a family the manifest
    lists (a ninth binding with no manifest entry fails here);
  * the names a binding registers in PyInit (`def_function[...]("name")`)
    are exactly the names the manifest lists for it, so a function exported
    but undeclared, or declared but unexported, fails;
  * every `bindings/build_<family>_host.sh` is the two-line shim over
    `bindings/build_host_family.sh` and names its own family;
  * no host binding reads DETECTED_COLUMN, and every one carries the
    build-time assert that it compiled as the kernel matrix's CPU column;
  * every covered training lane is a lane tools/identity_break.py defines,
    every inference lane is one tools/classical_host_gate.py knows, and the
    forest kinds are tools/forest_host_gate.py's KINDS;
  * every recording directory and GPU column the manifest names exists;
  * `_backend._HOST_MODULES` is the manifest's routing table, and the CPU
    identity gate workflow reads its lists from the manifest rather than
    carrying literals.
"""
import re
from pathlib import Path

import pytest

from mojolearn import host_surface

ROOT = Path(__file__).resolve().parents[3]
BINDINGS = ROOT / "bindings"

DEF_FUNCTION = re.compile(r'(?:module|m)\.def_function\[[A-Za-z0-9_]+\]\("([A-Za-z0-9_]+)"\)')


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _exports_in_source(name):
    return DEF_FUNCTION.findall(_read(host_surface.binding_source(name)))


@pytest.mark.parametrize("name", host_surface.families())
def test_family_has_a_binding_source(name):
    assert (ROOT / host_surface.binding_source(name)).is_file(), (
        f"the manifest lists family {name!r} but {host_surface.binding_source(name)} does not exist"
    )


def test_every_host_binding_source_is_a_family():
    on_disk = sorted(
        p.name[len("_mojolearn_"):-len("_host.mojo")]
        for p in BINDINGS.glob("_mojolearn_*_host.mojo")
    )
    assert on_disk == sorted(host_surface.families()), (
        "a host binding source exists that the manifest does not list, or the reverse"
    )


@pytest.mark.parametrize("name", host_surface.families())
def test_binding_exports_exactly_the_manifest(name):
    exported = _exports_in_source(name)
    assert exported, f"no def_function registrations found in {host_surface.binding_source(name)}"
    listed = list(host_surface.family(name)["exports"])
    assert sorted(exported) == sorted(listed), (
        f"{name}: binding exports {sorted(set(exported) - set(listed))} the manifest does not list; "
        f"manifest lists {sorted(set(listed) - set(exported))} the binding does not export"
    )
    assert len(exported) == len(set(exported)), f"{name}: a name is registered twice"


@pytest.mark.parametrize("name", host_surface.families())
def test_binding_reads_back_the_cpu_column(name):
    prefix = f"{name}_host"
    exports = set(host_surface.family(name)["exports"])
    for suffix in host_surface.READBACK:
        assert f"{prefix}_{suffix}" in exports, f"{name}: no {prefix}_{suffix} read-back export"
    source = _read(host_surface.binding_source(name))
    assert "DETECTED_COLUMN" not in source, f"{name}: a host binding must not read DETECTED_COLUMN"
    assert "comptime assert TARGET_COLUMN == COLUMN_CPU" in source, (
        f"{name}: the build-time assert that the binding compiled as COLUMN_CPU is missing"
    )


@pytest.mark.parametrize("name", host_surface.families())
def test_build_shim_execs_the_one_builder(name):
    shim = ROOT / host_surface.build_shim(name)
    assert shim.is_file(), f"{shim} is missing"
    lines = [line for line in shim.read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert len(lines) == 1, f"{shim.name} is not a one-command shim: {lines}"
    assert re.fullmatch(
        r'exec sh "\$\(dirname -- "\$0"\)/build_host_family\.sh" ' + re.escape(name) + r'( "\$@")?',
        lines[0],
    ), f"{shim.name} does not exec {host_surface.BUILDER} with its own family: {lines[0]!r}"
    assert host_surface.BUILDER == "bindings/build_host_family.sh"
    assert (ROOT / host_surface.BUILDER).is_file()


def test_no_second_builder():
    builders = sorted(p.name for p in BINDINGS.glob("build_*host*.sh") if p.name != "build_host_family.sh"
                      and not re.fullmatch(r"build_[a-z_]+_host\.sh", p.name))
    assert builders == [], f"a builder beside build_host_family.sh: {builders}"


def test_sabotage_define_reaches_each_binding():
    """The define the gate passes for a family's negative control must be
    read somewhere the binding compiles: its own source or a host module it
    names."""
    for f in host_surface.FAMILIES:
        define = f["sabotage_define"]
        texts = [_read(host_surface.binding_source(f["family"]))]
        texts += [_read(m) for m in f["host_modules"] if (ROOT / m).is_file()]
        assert any(define in t for t in texts), (
            f"{f['family']}: {define} is read by neither the binding nor its host modules"
        )


#: The fit entry points an inference-only binding must never name, by family
#: (the neighbors and density inference lane, 2026-09-15).
#: Mojo compiles only what a binding reaches, so a source that names none of
#: these ships none of them; bench/results/identity_break/
#: 2026-09-15_inference-iforest-gmm-hdbscan/fit_symbols.txt shows the built
#: files agree.
_FIT_NAMES = {
    "mixture_infer": ("gmmh_fit", "gmm_fit", "gmmh_m_step", "gmmh_initial_resp"),
    "hdbscan_infer": ("hdbh_fit", "hdbscan_fit", "generate_prediction_data"),
    "gp_infer": ("gpr_host_fit", "gpr_fit", "gpc_host_fit", "gpc_fit", "chol_host_potrf", "cholesky_factor"),
}


def _mojo_code(text):
    """Mojo source with its triple-quoted strings and `#` comments removed,
    so a docstring saying what a binding leaves out is not read as code."""
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return re.sub(r"#[^\n]*", "", text)


def test_inference_only_bindings_name_no_fit():
    for name, fits in _FIT_NAMES.items():
        f = host_surface.family(name)
        texts = [_mojo_code(_read(host_surface.binding_source(name)))]
        texts += [_mojo_code(_read(m)) for m in f["host_modules"] if m.startswith("bindings/")]
        for fit in fits:
            hits = [t for t in texts if re.search(rf"\b{fit}\b", t)]
            assert not hits, f"{name}: {fit} is named in its binding sources"


def test_host_modules_exist():
    for f in host_surface.FAMILIES:
        for m in f["host_modules"]:
            assert (ROOT / m).is_file(), f"{f['family']}: host module {m} does not exist"


def _loop_registered_lanes(text):
    """The lanes tools/identity_break.py registers in a module-level loop,
    `for _name, _nu, _ls in (("matern12", ...), ...): lane(f"gp-{_name}")(...)`,
    which the `@lane("...")` pattern cannot see (the gp Matern lanes, the kde
    kernel and metric pairs, the knn and radius metrics). Read from the AST:
    each literal tuple of the loop binds the target names, and the f-string
    is spelled out from the string constants it binds."""
    import ast

    out = set()
    for node in ast.parse(text).body:
        if not isinstance(node, ast.For) or not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        names = [t.id if isinstance(t, ast.Name) else None for t in targets]
        for stmt in node.body:
            call = getattr(stmt, "value", None)
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Call)
                    and isinstance(call.func.func, ast.Name) and call.func.func.id == "lane"
                    and call.func.args and isinstance(call.func.args[0], ast.JoinedStr)):
                continue
            for element in node.iter.elts:
                values = element.elts if isinstance(element, ast.Tuple) else [element]
                bound = {
                    n: v.value for n, v in zip(names, values)
                    if n and isinstance(v, ast.Constant) and isinstance(v.value, str)
                }
                parts = []
                for piece in call.func.args[0].values:
                    if isinstance(piece, ast.Constant):
                        parts.append(str(piece.value))
                    elif isinstance(piece, ast.FormattedValue) and isinstance(piece.value, ast.Name) \
                            and piece.value.id in bound:
                        parts.append(bound[piece.value.id])
                    else:
                        parts = None
                        break
                if parts is not None:
                    out.add("".join(parts))
    return out


def test_covered_lanes_are_identity_break_lanes():
    text = _read("tools/identity_break.py")
    defined = set(re.findall(r'^@lane\("([a-z0-9-]+)"\)', text, re.M))
    assert defined, "no @lane registrations found in tools/identity_break.py"
    looped = _loop_registered_lanes(text)
    assert {"gp-matern12", "gp-matern32", "gp-matern52-ard"} <= looped, (
        f"the loop reader no longer sees the gp Matern lanes: {sorted(looped)}"
    )
    defined |= looped
    missing = [lane for lane in host_surface.covered_lanes() if lane not in defined]
    assert missing == [], f"covered lanes unknown to tools/identity_break.py: {missing}"


def test_inference_lanes_are_classical_gate_lanes():
    text = _read("tools/classical_host_gate.py")
    m = re.search(r"^LANES = \{(.*?)^\}", text, re.S | re.M)
    assert m, "no LANES table in tools/classical_host_gate.py"
    known = set(re.findall(r"^\s+'([a-z0-9-]+)': \(", m.group(1), re.M))
    assert sorted(host_surface.inference_lanes()) == sorted(known), (
        f"manifest inference lanes {sorted(host_surface.inference_lanes())} and the classical gate's "
        f"LANES {sorted(known)} disagree"
    )


def test_forest_kinds_are_the_forest_gate_kinds():
    text = _read("tools/forest_host_gate.py")
    m = re.search(r"^KINDS = \{(.*?)^\}", text, re.S | re.M)
    assert m, "no KINDS table in tools/forest_host_gate.py"
    known = re.findall(r"^\s+'([a-z_]+)': dict\(", m.group(1), re.M)
    assert host_surface.forest_kinds() == known


def test_recordings_and_columns_exist():
    for rel in (host_surface.CLASSICAL_RECORDED + host_surface.FORECAST_RECORDED + host_surface.INFERENCE_ONLY_RECORDED
                + host_surface.SEARCH_LOOKUP_RECORDED
                + host_surface.CLASSICAL_GPU_COLUMNS
                + (host_surface.FOREST_RECORDED_ROOT,)):
        assert (ROOT / rel).exists(), f"the manifest names {rel}, which is not in the tree"


def test_training_gpu_columns_exist():
    """The record must not lag the surface: the training gate diffs the CPU
    column against these three files with --require-columns 4, and a column
    that is not in the tree is a gate that cannot run, not a pass."""
    missing = [rel for rel in host_surface.TRAINING_GPU_COLUMNS if not (ROOT / rel).exists()]
    assert missing == [], f"the manifest names training GPU columns not in the tree: {missing}"


def test_backend_routes_the_manifest():
    from mojolearn import _backend
    assert _backend._HOST_MODULES == host_surface.routed_modules()
    for gpu_family in host_surface.routed_modules():
        assert gpu_family in _backend._MODULES, f"{gpu_family} is routed but is not a _MODULES family"


def test_inference_routes_ship_and_carry_no_fit():
    """lane/inference-forecast-umap-pca (2026-09-15): an inference-only
    binding that serves a route on a CPU-only install ships, serves a real
    `_MODULES` family whose reference binding does not ship, registers the
    reference binding's prediction names and no `*_fit` name, and
    `_backend` reads the table from the manifest."""
    from mojolearn import _backend
    routes = host_surface.inference_routes()
    assert routes == {
        "_mojolearn_arima": "_mojolearn_forecast_host",
        # lane/inference-embedding-ivf-cholesky (2026-09-15)
        "_mojolearn_ivf": "_mojolearn_ivf_search_host",
        "_mojolearn_embedding": "_mojolearn_embedding_infer_host",
    }
    assert _backend._HOST_INFERENCE_MODULES == routes
    shipped = set(host_surface.wheel_bindings())
    for route, binding in routes.items():
        assert route in _backend._MODULES
        assert binding in shipped
        reference = host_surface.routed_modules().get(route)
        assert reference and reference not in shipped, f"{route}: the reference binding ships; no fallback is needed"
        name = binding[len("_mojolearn_"):-len("_host")]
        exported = _exports_in_source(name)
        assert not [e for e in exported if e.endswith("_fit")], f"{binding} registers a fit: {exported}"
        ref_exports = set(host_surface.family(reference[len("_mojolearn_"):-len("_host")])["exports"])
        served = [e for e in exported if not e.startswith(name + "_host_")]
        assert set(served) <= ref_exports, f"{binding} registers names its reference binding does not"


def test_workflow_reads_the_manifest_not_literals():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    for var in ("COVERED_LANES", "BUILD_FAMILIES", "SABOTAGE_FAMILIES", "ALL_HOST_BINDINGS",
                "CLASSICAL_RECORDED", "SAVED_MODEL_RECORDED"):
        assert not re.search(rf'^\s+{var}: "', text, re.M), (
            f"the workflow carries a literal {var}; it must read the manifest"
        )
    assert not re.search(r"^\s+GPU_COLUMNS: >-", text, re.M), (
        "the workflow carries a literal GPU_COLUMNS block; it must read the manifest"
    )
    assert "python/mojolearn/host_surface.py" in text
    for flag in ("--covered-lanes", "--families", "--wheel-families", "--bindings", "--wheel-bindings",
                 "--classical-recorded", "--saved-model-recorded", "--classical-gpu-columns",
                 "--training-gpu-columns", "cpu_identity_gate_check.py build-list"):
        assert flag in text, f"the workflow does not read {flag} from the manifest"
    for rel in host_surface.TRAINING_GPU_COLUMNS + host_surface.CLASSICAL_GPU_COLUMNS:
        directory = "/" + rel.rsplit("/", 1)[0] + "/"
        assert directory in text, f"the sparse checkout does not bring down {directory}"
    # Every build loop reads the manifest's list; no binding is built by hand
    # (until 2026-09-15 byte_lm, forest and tokenizer were, and seven shipped
    # bindings were never built).
    # The one hand build left is the forest sabotage binding, which reads its
    # own define (MOJOLEARN_FOREST_HOST_SABOTAGE) into its own directory.
    assert re.findall(r"sh bindings/build_([a-z_]+)_host\.sh", text) == ["forest"], "the workflow builds a family by hand"
    assert text.count('for family in $BUILD_FAMILIES; do') == 1
    assert text.count('for family in $SABOTAGE_FAMILIES; do') == 1
    for rel in host_surface.saved_model_recorded():
        assert rel.startswith("bench/results/classical_host/"), rel
    assert "/bench/results/classical_host/" in text


def test_public_inference_bindings_ship_and_packaging_reads_the_manifest():
    """Inference dependencies ship; reference-only routes remain source-built.
    Builders, packers and installed-wheel checks share the manifest.
    """
    from mojolearn import _backend
    shipped = set(host_surface.wheel_bindings())
    assert set(_backend._HOST_MODULES.values()) <= set(host_surface.bindings())
    assert {"_mojolearn_byte_lm_host", "_mojolearn_forest_host", "_mojolearn_tokenizer_host",
            "_mojolearn_neural_host"} <= shipped
    assert set(host_surface.wheel_families()) == {
        "byte_lm", "forest", "tokenizer", "neural", "core", "linalg", "estimators", "metrics", "svm", "forecast",
        "mixture_infer", "hdbscan_infer", "gp_infer",
        "embedding_infer", "ivf_search",
    }
    # The inference-only families ship; the reference families whose
    # scoring and prediction entries they carry do not.
    for inference, reference in (("mixture_infer", "mixture"), ("hdbscan_infer", "hdbscan"), ("gp_infer", "gp")):
        assert host_surface.family(inference)["ships_in_wheel"] and host_surface.family(inference)["routes"] is None
        assert not host_surface.family(reference)["ships_in_wheel"]
        assert host_surface.family(inference)["training_lanes"] == ()
    assert len(host_surface.families()) > len(host_surface.wheel_families())
    assert host_surface.training_gpu_column_record() == host_surface.TRAINING_GPU_COLUMNS[0].rsplit("/", 2)[1]
    for rel, token in (
        ("packaging/linux/pack_wheel.py", "wheel_host_bindings()"),
        ("packaging/linux/build_sets.sh", "host_surface.py --wheel-families"),
        ("packaging/macos/build_release_wheel.sh", "host_surface.py --wheel-families"),
        ("packaging/linux/smoke.py", "host_surface.wheel_bindings()"),
        ("packaging/macos/verify_wheel.sh", "host_surface.wheel_bindings()"),
        ("tools/linux_surface_qualification.sh", "wheel_host_bindings()"),
        ("tools/check_linux_release_qualification.py", "wheel_host_bindings()"),
    ):
        text = _read(rel)
        assert token in text, f"{rel} does not read the host list from the manifest by {token!r}"
        assert not re.search(r"^\s*HOST_NAMES?\s*=\s*[\"'(]\s*\"?_mojolearn_[a-z_]+_host", text, re.M), (
            f"{rel} carries a host name list of its own"
        )


def test_command_line_agrees_with_the_api(capsys):
    assert host_surface.main(["--covered-lanes"]) == 0
    assert capsys.readouterr().out.strip() == ",".join(host_surface.covered_lanes())
    assert host_surface.main(["--routed-families"]) == 0
    assert capsys.readouterr().out.strip() == " ".join(host_surface.routed_families())
    assert host_surface.main(["--bindings", "--sep", ","]) == 0
    assert capsys.readouterr().out.strip() == ",".join(host_surface.bindings())


def test_training_lane_names_cover_every_covered_lane():
    missing = [lane for lane in host_surface.covered_lanes() if lane not in host_surface.TRAINING_LANE_NAMES]
    assert missing == [], f"no doc name for covered lanes {missing}"
