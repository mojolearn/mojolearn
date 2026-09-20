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
import json
import re
from pathlib import Path

import pytest

from mojolearn import host_surface

ROOT = Path(__file__).resolve().parents[3]
BINDINGS = ROOT / "bindings"

#: The three vendor classes a cross-vendor claim needs, as the shipped
#: table's `cols` spells them. `cpu` is deliberately not one of them: it is
#: the column the user's own machine reproduces, so it cannot also be one of
#: the independent witnesses that make the reference worth reproducing.
TRAINING_GPU_CLASSES = ("amd", "apple", "nvidia")

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
    # lane/inference-holtwinters (2026-09-15): the Holt-Winters fit, its
    # decomposition and BFGS, and the ARIMA fit.
    "forecast": ("holtwinters_fit", "oracle_fit", "holtwinters_validate_params", "host_r1qt",
                 "arima_fit", "oracle_eval"),
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
    # Module-level call registrations, `lane("gmm-sample")(_gmm_sample_lane("kmeans"))`
    # (the sample and sample_y lanes; lane/cpu-verifier-gaps-7, 2026-09-15).
    called = set(re.findall(r'^lane\("([a-z0-9-]+)"\)\(', text, re.M))
    assert {"gmm-sample", "gp-sample-y"} <= called, f"the call reader no longer sees the sample lanes: {sorted(called)}"
    defined |= called
    looped = _loop_registered_lanes(text)
    assert {"gp-matern12", "gp-matern32", "gp-matern52-ard"} <= looped, (
        f"the loop reader no longer sees the gp Matern lanes: {sorted(looped)}"
    )
    defined |= looped
    missing = [lane for lane in host_surface.covered_lanes() if lane not in defined]
    assert missing == [], f"covered lanes unknown to tools/identity_break.py: {missing}"


def test_inference_lanes_are_classical_gate_lanes():
    # Evaluate the binding-free registry so generated kernel families are
    # counted too; a regex over literal keys silently misses those APIs.
    import runpy
    known = set(runpy.run_path(str(ROOT / "tools/classical_host_gate.py"))["LANES"])
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
        # lane/inference-holtwinters (2026-09-15)
        "_mojolearn_tsa": "_mojolearn_forecast_host",
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
        assert reference, f"{route} is served by {binding} but is not a routed family"
        name = binding[len("_mojolearn_"):-len("_host")]
        exported = _exports_in_source(name)
        assert not [e for e in exported if e.endswith("_fit")], f"{binding} registers a fit: {exported}"
        # One inference binding may serve several routes (forecast serves
        # ARIMA and Holt-Winters): every name it registers is a name of one
        # of the reference bindings of the routes it serves.
        ref_exports = set()
        for other, b in routes.items():
            if b == binding:
                ref = host_surface.routed_modules()[other]
                ref_exports |= set(host_surface.family(ref[len("_mojolearn_"):-len("_host")])["exports"])
        served = [e for e in exported if not e.startswith(name + "_host_")]
        assert set(served) <= ref_exports, f"{binding} registers names its reference bindings do not"


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
    held_back = [f["family"] for f in host_surface.FAMILIES if not f["ships_in_wheel"]]
    assert held_back == [], (
        f"these families do not ship: {held_back}. Every host family ships since 2026-09-16 "
        "(lane/ship-cpu-host-families), so a wheel can check every lane whose reference is "
        "current; holding one back makes its lanes unverifiable on an installed CPU and needs a "
        "reason written into its wheel_note"
    )
    assert host_surface.wheel_families() == host_surface.families()
    # The inference-only families ship; the reference families whose
    # scoring and prediction entries they carry do not.
    for inference, reference in (("mixture_infer", "mixture"), ("hdbscan_infer", "hdbscan"), ("gp_infer", "gp")):
        assert host_surface.family(inference)["ships_in_wheel"] and host_surface.family(inference)["routes"] is None
        assert host_surface.family(reference)["ships_in_wheel"], "every family ships since 2026-09-16"
        assert host_surface.family(inference)["training_lanes"] == ()
    assert set(host_surface.wheel_bindings()) == set(host_surface.bindings())
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


#: lane/cpu-verifier-gaps-7 (2026-09-15): the seven one-device lanes that had a
#: CPU host function and no gate wiring, by the family whose binding computes
#: their CPU cells.
GAPS_7 = {
    "gmm-sample": "mixture", "gmm-random-init-sample": "mixture",
    "gp-sample-y": "gp", "gp-sample-y-normalize": "gp",
    "tokenizer": "tokenizer",
    "gbdt-categorical-ctr-tables": "forest", "gbdt-tensor-ctr-tables": "forest",
}


def test_gaps_7_are_covered_by_their_families():
    covered = host_surface.covered_lanes()
    record = host_surface.record_covered_lanes()
    for lane, fam in GAPS_7.items():
        assert lane in covered, f"{lane} is not a covered lane"
        assert lane in record, f"{lane} is not diffed against the training GPU columns (OWED there)"
        assert lane in host_surface.family(fam)["training_lanes"], f"{lane} is not declared by {fam}"


def test_gate_sabotage_defines_reach_the_tokenizer_and_the_ctr_arm():
    """MOJOLEARN_HOST_SABOTAGE reaches nothing in the tokenizer binding and
    only the CTR arm moves a CTR table lane without touching every other
    forest prediction, so the gate's sabotage set builds both with their own
    define, and the define must exist in the binding source."""
    # Two own defines for the tokenizer since lane/laneless-public-classes
    # (2026-09-19): the encoder arm leaves bpe-trainer where it found it, so
    # the trainer's own arm is in the set too (GATE_SABOTAGE_OWN_DEFINES).
    assert host_surface.sabotage_build_defines("tokenizer").split() == [
        "-D", "MOJOLEARN_HOST_SABOTAGE=1", "-D", "MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1",
        "-D", "MOJOLEARN_BPE_TRAINER_SABOTAGE=1"]
    assert host_surface.sabotage_build_defines("forest").split() == [
        "-D", "MOJOLEARN_HOST_SABOTAGE=1", "-D", "MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1"]
    # lane/catboost-parity: the non-default border types' own arm
    assert host_surface.sabotage_build_defines("gbdt").split() == [
        "-D", "MOJOLEARN_HOST_SABOTAGE=1", "-D", "MOJOLEARN_BORDER_TYPES_SABOTAGE=1",
        "-D", "MOJOLEARN_ORDERED_SABOTAGE=1",
        "-D", "MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE=1"]
    for name in host_surface.families():
        if name not in host_surface.GATE_SABOTAGE_OWN_DEFINES:
            assert host_surface.sabotage_build_defines(name) == "-D MOJOLEARN_HOST_SABOTAGE=1", name
    # The linalg family joined GATE_SABOTAGE_OWN_DEFINES on
    # lane/laneless-public-classes (2026-09-19): MOJOLEARN_HOST_SABOTAGE
    # reaches gemm_oracle's leaf and gemm_int8_oracle's dequantized cell and
    # NOTHING in the four conversion seams of contract L-1..L-6, measured --
    # the lowbit-conversions cell read the clean hash on all nine fixtures
    # under the family define alone.
    assert host_surface.sabotage_build_defines("linalg").split() == [
        "-D", "MOJOLEARN_HOST_SABOTAGE=1", "-D", "MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1"]
    sources = {"tokenizer": ["bindings/_mojolearn_tokenizer_host.mojo"],
               "forest": ["core/gbdt_host_ctr.mojo", "bindings/_mojolearn_forest_host.mojo"],
               "linalg": ["gemm/host/gemm_lowbit_oracle.mojo"]}
    sources["tokenizer"].append("tokenizer/train/bpe_train.mojo")
    sources["gbdt"] = ["gbdt/grid_creator/binarization.mojo",
                       "gbdt/host/gbdt_oracle_ordered.mojo",
                       "gbdt/metrics/sample_quantile.mojo"]
    for name, defines in host_surface.GATE_SABOTAGE_OWN_DEFINES.items():
        text = "".join(_read(rel) for rel in sources[name])
        for define in defines:
            # WHITESPACE-TOLERANT ON PURPOSE (2026-09-19). This read
            # `f'is_defined["{define}"]' in text` and went red on
            # MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE when a perf commit wrapped
            # the very same call across two lines:
            #
            #     comptime SAMPLE_QUANTILE_SABOTAGE = is_defined[
            #         "MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE"
            #     ]()
            #
            # The switch was intact and the arm still fired; the substring
            # was what broke. A check on a STRING SPELLING is not a check on
            # the PROPERTY, and this one failed in the safe direction only by
            # luck -- the same brittleness would pass a define that had been
            # commented out on one line.
            assert re.search(r'is_defined\s*\[\s*"%s"\s*\]' % re.escape(define), text), (
                f"{define} is read by no {name} source")
    with pytest.raises(KeyError):
        host_surface.sabotage_build_defines("nonesuch")


def test_gbdt_ctr_models_cover_every_fixture():
    """The CTR table lanes' CPU cells load one Metal-saved model per lane and
    fixture; a missing file reads REFUSED on the CPU column and fails the gate."""
    text = _read("tools/identity_break.py")
    m = re.search(r"^FIXTURES = \[(.*?)\]", text, re.M)
    assert m, "no FIXTURES list in tools/identity_break.py"
    fixtures = re.findall(r'"([a-z_]+)"', m.group(1))
    assert len(fixtures) == 9, fixtures
    assert 'GBDT_CTR_MODELS_ENV = "MOJOLEARN_IDENTITY_GBDT_CTR_MODELS"' in text
    d = ROOT / host_surface.GBDT_CTR_MODELS_DIR
    missing = [f"{lane}.{fx}.npz" for lane in host_surface.GBDT_CTR_MODEL_LANES for fx in fixtures
               if not (d / f"{lane}.{fx}.npz").is_file()]
    assert missing == [], f"{d} lacks {missing}"
    # Every CTR lane is a forest training lane. The reverse stopped holding on
    # lane/laneless-public-classes (2026-09-19), when `saved-model-host-infer`
    # joined the family: that lane FITS on the fixture and saves, so it needs
    # no bundled model, and an equality here would have forced a second
    # meaning onto this tuple.
    assert set(host_surface.GBDT_CTR_MODEL_LANES) <= set(host_surface.family("forest")["training_lanes"])
    assert set(host_surface.family("forest")["training_lanes"]) - set(host_surface.GBDT_CTR_MODEL_LANES) \
        == {"saved-model-host-infer"}


def test_workflow_wires_the_gaps_7_lanes():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    assert "/" + host_surface.GBDT_CTR_MODELS_DIR.rsplit("/", 1)[0] + "/" in text, "the models are not checked out"
    assert "MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$GITHUB_WORKSPACE/$(python3 $manifest --gbdt-ctr-models)" in text
    assert '--sabotage-build-defines "$family"' in text
    assert 'MOJOLEARN_BUILD_EXTRA_DEFINES: "-D MOJOLEARN_HOST_SABOTAGE=1"' not in text, (
        "a step-level sabotage define list would override the manifest's per-family defines")
    sab_run = text.split("covered lanes must diverge under MOJOLEARN_HOST_SABOTAGE", 1)[1].split("- name:", 1)[0]
    assert 'MOJOLEARN_FOREST_HOST_BINARY="${{ runner.temp }}/host-sab/_mojolearn_forest_host.so"' in sab_run
    assert "MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1" in sab_run


def test_command_line_prints_the_gaps_7_wiring(capsys):
    assert host_surface.main(["--gbdt-ctr-models"]) == 0
    assert capsys.readouterr().out.strip() == host_surface.GBDT_CTR_MODELS_DIR
    assert host_surface.main(["--sabotage-build-defines", "tokenizer"]) == 0
    assert capsys.readouterr().out.strip() == host_surface.sabotage_build_defines("tokenizer")


# ------------------------------------------------- the exposure surface
# lane/expose-inference-surface (2026-09-16): a family that does not ship,
# and a surface that is built but not reachable, must say so IN THE FILE.


def test_every_family_says_why_it_ships_or_does_not():
    """No silent exclusion: every family carries a `wheel_note`, and a family
    that does not ship says what serves its inference instead, or that it is
    training-only, or that the question is open."""
    missing = [f["family"] for f in host_surface.FAMILIES if not (f.get("wheel_note") or "").strip()]
    assert missing == [], f"families with no wheel_note: {missing}"
    assert set(host_surface.wheel_notes()) == set(host_surface.families())
    for f in host_surface.FAMILIES:
        note = f["wheel_note"]
        assert len(note) > 60, f"{f['family']}: the wheel_note is too short to be a reason: {note!r}"
        if f["ships_in_wheel"]:
            assert note.startswith("Ships:"), f"{f['family']}: a shipping family's note must start 'Ships:'"
        else:
            assert note.startswith("Does not ship"), (
                f"{f['family']}: a non-shipping family's note must start 'Does not ship'")


def test_the_analysis_functions_are_reachable_on_a_cpu_only_install():
    """Andrew's call (2026-09-16): the inference boundary keeps CPU TRAINING
    OF MODELS internal; it was never meant to exclude analysis functions that
    compute a statistic from the caller's own data. bootstrap, the permutation
    test, Monte Carlo integration and kpss_test train no model, so each must be
    reachable from a shipped binding rather than refusing on a laptop."""
    notes = host_surface.wheel_notes()
    assert "OPEN" not in notes["resample"] and "OPEN" not in notes["tsa"], (
        "these were decided; a note still reading OPEN would misreport the decision")
    assert notes["tsa"].startswith("Ships:"), "tsa ships since lane/ship-cpu-host-families"

    # resample ships, and its binding registers the three entries and no fit.
    resample = host_surface.family("resample")
    assert resample["ships_in_wheel"], "resample must ship; its functions train no model"
    assert "_mojolearn_resample_host" in host_surface.wheel_bindings()
    exported = _exports_in_source("resample")
    for name in ("bootstrap", "permutation_test", "monte_carlo_integrate"):
        assert name in exported, f"the resample binding does not register {name}"
    assert [e for e in exported if e.endswith("_fit")] == [], "the resample binding registers a fit"

    # kpss_test is served by the SHIPPED forecast binding. Since
    # lane/ship-cpu-host-families (2026-09-16) the tsa family ships as well, so
    # that is no longer the ONLY way to reach kpss_test on a CPU -- but it is
    # still the one this test pins, because it is what keeps a saved
    # Holt-Winters model answering from a binary with no fit in it.
    assert host_surface.family("tsa")["ships_in_wheel"], "every family ships since 2026-09-16"
    forecast = host_surface.family("forecast")
    assert forecast["ships_in_wheel"] and "kpss_test" in forecast["exports"]
    assert "kpss_test" in _exports_in_source("forecast"), (
        "the shipped forecast binding does not register kpss_test")
    assert host_surface.inference_routes()["_mojolearn_tsa"] == "_mojolearn_forecast_host"

    # ONE SOURCE: both bindings register kpss_test from the shared module, so
    # the reference binary and the shipped binary cannot drift apart.
    shared = "bindings/kpss_host_test.mojo"
    assert (ROOT / shared).is_file(), f"{shared} is missing"
    for binding in ("_mojolearn_tsa_host", "_mojolearn_forecast_host"):
        src = _read(f"bindings/{binding}.mojo")
        assert "from bindings.kpss_host_test import" in src, (
            f"{binding} does not register kpss_test from the shared module")
    assert "def kpss_test_binding(" not in _read("bindings/_mojolearn_tsa_host.mojo"), (
        "the tsa binding still defines its own kpss_test_binding; the two binaries would drift")
    assert shared in forecast["host_modules"]


def test_promoted_reference_lanes_still_carry_the_classes_that_promoted_them():
    """The eighteen admitted 2026-09-19 are public, and the shipped table still
    shows the evidence that made them public.

    A promotion is a claim an installed wheel makes to a user: `verify --all`
    on a CPU-only box selects these lanes and compares them against the
    bundled table. If a later table edit drops a vendor class from one part of
    one fixture, the claim stops being true and nothing else in this file would
    notice -- the candidate assertion cannot, because the list it reads is now
    empty. This is that missing direction, and it reads the SHIPPED TABLE for
    the same reason the candidate check did: a record in the tree is not what
    an install checks against.

    Intersected over parts, never unioned. A union would pass on a lane whose
    nine fixtures each rest on a different single column.
    """
    promoted = host_surface.PUBLIC_REFERENCE_PROMOTED
    assert promoted, "the promoted list is empty"
    assert len(set(promoted)) == len(promoted), "a lane is named twice"

    live = set(host_surface.public_reference_lanes())
    missing = [lane for lane in promoted if lane not in live]
    assert missing == [], (
        "promoted lanes that public_reference_lanes() no longer returns, so the wheel stopped "
        f"checking what it was promoted for: {missing}")

    assert not (set(promoted) & set(host_surface.public_reference_candidates())), (
        "a lane is both promoted and still a candidate")

    table = json.loads(_read("python/mojolearn/verify_reference/table.json"))
    required = set(TRAINING_GPU_CLASSES)
    short = {}
    for lane in promoted:
        have = _classes_on_every_part(table, lane)
        if not required <= have:
            short[lane] = sorted(required - have)
    assert short == {}, (
        "the shipped table no longer carries every vendor class on every part of every fixture for "
        "these promoted lanes, so an installed wheel is making a public identity claim the bundled "
        "evidence does not support. Either restore the columns or demote the lane:\n  "
        + "\n  ".join(f"{lane}: missing {', '.join(cls)}" for lane, cls in sorted(short.items())))


def test_public_reference_candidates_meet_every_condition_to_be_promoted():
    """Each candidate is a real lane, is diffed against the record columns,
    is served only by shipping families, and is not already live. A candidate
    that fails one of these could not be promoted by the run that is owed."""
    candidates = host_surface.public_reference_candidates()
    assert len(set(candidates)) == len(candidates), "a lane is named twice"
    if not candidates:
        # EMPTY IS THIS MECHANISM'S SUCCESS STATE (2026-09-19): every lane that
        # was waiting on a column got one. It is not "nothing to check" -- the
        # checking moves to the promoted set below, which is what keeps the
        # admission falsifiable. Without this, emptying the list would silently
        # retire the only assertion standing behind eighteen public claims.
        return

    text = _read("tools/identity_break.py")
    defined = set(re.findall(r'^@lane\("([a-z0-9-]+)"\)', text, re.M))
    defined |= set(re.findall(r'^lane\("([a-z0-9-]+)"\)\(', text, re.M))
    defined |= _loop_registered_lanes(text)
    unknown = [l for l in candidates if l not in defined]
    assert unknown == [], f"candidates unknown to tools/identity_break.py: {unknown}"

    live = set(host_surface.public_reference_lanes())
    already = [l for l in candidates if l in live]
    assert already == [], f"candidates that are already public reference lanes: {already}"

    record = set(host_surface.record_covered_lanes())
    outside = [l for l in candidates if l not in record]
    assert outside == [], f"candidates not diffed against TRAINING_GPU_COLUMNS: {outside}"

    # Reachable from a SHIPPED binding, so promoting adds no binary to the
    # wheel. Two ways to be reachable, and the second is not a loophole: a
    # family that holds a fit stays a source build while a shipped
    # inference-only binding serves its route (kpss through forecast, since
    # lane/expose-inference-surface). Requiring the declaring family itself to
    # ship would have rejected kpss, which a user can in fact call.
    shipped_bindings = set(host_surface.wheel_bindings())
    routes = host_surface.inference_routes()
    for lane in candidates:
        serving = [f for f in host_surface.FAMILIES if lane in f["training_lanes"]]
        assert serving, f"{lane}: no family declares it as a training lane"
        for f in serving:
            if f["ships_in_wheel"]:
                continue
            served_by = routes.get(f["routes"])
            assert served_by in shipped_bindings, (
                f"{lane}: declared by {f['family']}, which does not ship, and no shipped binding "
                f"serves its route {f['routes']}; promoting it would grow the wheel")

    # THE CONDITION THAT ACTUALLY HOLDS THIS LIST, ASKED OF THE SHIPPED TABLE
    # (lane/lm-attention-fallback, 2026-09-19). Every other condition above is
    # checked; the three-vendor-class one -- the ONE every entry is in fact
    # waiting on -- was prose in the docstring, and prose went stale without
    # anything noticing. Measured this morning against the shipped table: the
    # `gp-optimize` comment said "only Apple GPU witnesses", the nine from
    # lane/ship-cpu-host-families said "rests on the APPLE column alone, with
    # no NVIDIA and no AMD", and `svc-poly` said "apple and cpu" -- and all
    # eighteen carry amd + apple + cpu today, because the 2026-09-18 AMD
    # columns landed and no one re-read the comments. A FLOOR IN PROSE IS NOT
    # A FLOOR: the reason has to travel with the number, in code a change
    # walks past.
    #
    # It is the promoting direction that is load-bearing. A candidate that
    # REACHES all three classes has nothing left holding it here, and leaving
    # it would understate the public surface by exactly the lanes a rented
    # column just paid for; this fails on that day and names them. The
    # converse is already covered: `already` above fails if one is promoted
    # early. Note what the assertion reads -- the SHIPPED TABLE, not the
    # record files in the tree. A column can be committed, admissible and
    # still absent from every installed verifier's copy until a scoped
    # `verify --emit-reference --reference-table` admits it, and it is the
    # installed copy a promotion promises against.
    table = json.loads(_read("python/mojolearn/verify_reference/table.json"))
    reached = {lane: sorted(_classes_on_every_part(table, lane))
               for lane in candidates
               if set(TRAINING_GPU_CLASSES) <= _classes_on_every_part(table, lane)}
    assert reached == {}, (
        "these candidates now carry all three vendor classes on every part of every fixture in the "
        "shipped table, so the condition that held them here is met: promote them into "
        "public_reference_lanes() (and drop any matching SAVED_MODEL_INFERENCE_OWED debt that the "
        "same column discharged):\n  "
        + "\n  ".join(f"{lane}: {', '.join(classes)}" for lane, classes in sorted(reached.items())))


def test_saved_model_inference_owed_names_real_lanes_and_remaining_debt():
    """Recording support and remaining vendor qualification are distinct.

    A declared lane may retain a debt after its first GPU recording, provided
    that real retained models exist and the outstanding qualification is named.
    """
    owed = host_surface.saved_model_inference_owed()
    declared = set(host_surface.inference_lanes())
    for lane in set(owed) & declared:
        assert "qualification remain owed" in owed[lane], lane
        record = ROOT / "bench/results/classical_host/2026-09-18-apple-kernel-variants/saved-models" / lane / lane
        assert len(list(record.glob("*/expected.json"))) == 9, lane
        assert len(list(record.glob("*/model.npz"))) == 9, lane
    text = _read("tools/identity_break.py")
    defined = set(re.findall(r'^@lane\("([a-z0-9-]+)"\)', text, re.M))
    defined |= set(re.findall(r'^lane\("([a-z0-9-]+)"\)\(', text, re.M))
    defined |= _loop_registered_lanes(text)
    host_src = _read("python/mojolearn/_classical_host.py")
    # The format STRINGS are defined in the impl modules (density.py,
    # _spectral_impl.py, _hierarchy_impl.py) and _classical_host.py dispatches
    # them by SYMBOL. host_surface.py is excluded on purpose: it carries these
    # notes, and a search that found its own text would pass whatever it said.
    package = ROOT / "python" / "mojolearn"
    defines = "\n".join(p.read_text(encoding="utf-8") for p in sorted(package.glob("*.py"))
                        if p.name != "host_surface.py")
    for lane, why in owed.items():
        assert lane in defined, f"{lane}: not a lane tools/identity_break.py defines"
        assert len(why) > 60, f"{lane}: the reason is too short: {why!r}"
        fmt = re.search(r"mojolearn-[a-z]+-\d", why)
        if fmt:
            assert f'"{fmt.group(0)}"' in defines, (
                f"{lane}: the note names {fmt.group(0)}, which no module under python/mojolearn defines")
            symbol = "_" + fmt.group(0).split("-")[1].upper() + "_FORMAT"
            assert re.search(rf"^\s+{symbol}: \{{", host_src, re.M), (
                f"{lane}: _classical_host._FORMATS does not dispatch {symbol}, so mojolearn.host_model "
                f"would refuse such a file and the note's claim is wrong")
        else:
            assert "no `save`" in why or "serialization format" in why, (
                f"{lane}: names no dispatched format and does not say why there is none")


def test_command_line_prints_the_exposure_surface(capsys):
    assert host_surface.main(["--public-reference-lanes"]) == 0
    assert capsys.readouterr().out.strip() == ",".join(host_surface.public_reference_lanes())
    assert host_surface.main(["--public-reference-candidates"]) == 0
    assert capsys.readouterr().out.strip() == ",".join(host_surface.public_reference_candidates())
    assert host_surface.main(["--wheel-notes"]) == 0
    assert "resample:" in capsys.readouterr().out
    assert host_surface.main(["--saved-model-inference-owed"]) == 0
    expected = "\n".join(f"{lane}: {reason}" for lane, reason in
                         host_surface.saved_model_inference_owed().items())
    assert capsys.readouterr().out.strip() == expected


def _reference_classes(table, lane):
    """The device classes the shipped table's cells for `lane` rest on.

    A cell part's `cols` maps a device class to the record its value came
    from, so the union over the lane's cells is every class that has ever
    been recorded reproducing it. One class means the reference has ONE
    witness: nothing has reproduced it, and a user who disagreed with it
    could not tell their machine apart from our single column.
    """
    classes = set()
    for key, cell in table["cells"].items():
        if key.partition("/")[0] != lane:
            continue
        for entry in cell.values():
            if entry.get("ref") is None or entry.get("conflict"):
                continue
            classes.update(entry.get("cols", {}))
    return classes


def _classes_on_every_part(table, lane):
    """The device classes that carry EVERY real part of EVERY fixture the
    shipped table has for `lane`, which is the strong form of the same
    question `_reference_classes` asks loosely.

    The union is the right reading for "has anything ever reproduced this
    lane at all", which is what the `one column` hold means. It is the WRONG
    reading for promotion: a lane can read three classes in the union while a
    single vendor carries `base` alone and the other eight fixtures rest on
    one column each, and promoting on that would claim a cross-vendor
    agreement over cells that never had one. Measured 2026-09-19, the
    difference is real and not hypothetical -- `gbdt-query-rmse`,
    `gmm-sample` and `gmm-random-init-sample` each carry apple in the union
    and lack it on 24, 15 and 15 of their parts respectively.

    Returns the empty set for a lane the table has no real part for, so a
    lane with no cells can never satisfy a caller asking for three classes.
    """
    per_part = []
    for key, cell in table["cells"].items():
        if key.partition("/")[0] != lane:
            continue
        for entry in cell.values():
            ref = entry.get("ref")
            if ref is None or str(ref).startswith("n/a") or entry.get("conflict"):
                continue
            per_part.append(set(entry.get("cols", {})))
    return set.intersection(*per_part) if per_part else set()


def _lane_revisions():
    """`LANE_REVISIONS` read from the harness's source, like the lane readers
    above, so this needs no numpy and no import of the harness."""
    import ast

    for node in ast.parse(_read("tools/identity_break.py")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LANE_REVISIONS" for t in node.targets):
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError("no LANE_REVISIONS in tools/identity_break.py")


def test_public_reference_lanes_are_derived_and_every_pending_reason_is_true():
    """THE PUBLIC SET IS DERIVED, AND THE HELD-BACK LIST IS CHECKED
    (lane/ship-cpu-host-families, 2026-09-16).

    `public_reference_lanes()` is every covered lane minus the `par-*`
    drivers, minus `PUBLIC_PENDING_LANES`, minus the candidates still waiting
    on a condition of their own. A pending list is a memo unless something
    holds it to the thing that made it true, so each reason is checked against
    its source: `stale reference` against the harness's LANE_REVISIONS,
    `no reference` against the shipped table, `own record` against
    TRAINING_FIX_LANES, and `measured` against the run recorded in the lane
    status, which is the only reason that comes from a run rather than from a
    static condition.

    The load-bearing direction is the last assertion: a lane whose fixture
    moves and which nobody adds here would become public carrying a reference
    that describes different bytes, and a user would read DIVERGENT for
    something that is not their machine."""
    import json

    table = json.loads(_read("python/mojolearn/verify_reference/table.json"))
    with_cells = {key.partition("/")[0] for key in table["cells"]}
    revisions = _lane_revisions()
    covered = set(host_surface.covered_lanes())
    host_only = set(host_surface.PUBLIC_HOST_ONLY_LANES)
    public = host_surface.public_reference_lanes()

    assert len(public) == len(set(public)), "a lane is listed twice"
    assert not [lane for lane in public if lane.startswith(host_surface.PUBLIC_EXCLUDED_PREFIXES)]
    assert not (set(public) & set(host_surface.PUBLIC_PENDING_LANES)), "a pending lane is public"
    trained = set(public) - host_only
    assert trained <= covered, f"public lanes with no CPU training path: {sorted(trained - covered)}"
    assert trained <= with_cells, f"public lanes the shipped table has no cell for: {sorted(trained - with_cells)}"
    recorded = set(host_surface.record_covered_lanes()) | set(host_surface.fix_covered_lanes())
    assert trained <= recorded, f"public lanes without a recorded route: {sorted(trained - recorded)}"

    # THE TABLE'S OWN REVISIONS, not merely "this lane has a revision entry"
    # (lane/inference-coverage-complete, 2026-09-16). `stale` means the
    # reference DESCRIBES DIFFERENT BYTES from the ones this harness makes,
    # which is a statement about the shipped table, and the table records the
    # revisions it was generated against. Asking only `lane in revisions`
    # cannot see a regeneration: a lane that gains a LANE_REVISIONS entry
    # would be barred from the public set FOREVER, because that entry never
    # goes away and the last assertion below then requires it to stay
    # pending. This is the same rule `_verify_reference.stale_reference_lanes`
    # applies at verify time, restated here against the source so the manifest
    # and the tool cannot drift apart. A table generated before the
    # `lane_revisions` key existed carries none, which is not evidence that it
    # is current, so it counts as stale.
    table_revisions = table.get("lane_revisions") or {}
    stale = {lane for lane, rev in revisions.items()
             if lane in with_cells and table_revisions.get(lane) != rev}

    wrong_reason = []
    for lane, why in host_surface.PUBLIC_PENDING_LANES.items():
        assert lane in covered, f"{lane} is held back but is not a covered lane at all"
        if why == "stale reference":
            assert lane in revisions, f"{lane}: no LANE_REVISIONS entry, so its reference is not stale; let it in"
            # Three outcomes, and the reason must name the right one. When a
            # release regenerates the table, a lane either keeps cells at the
            # current revision (promote it, once a run has been watched to
            # read IDENTICAL for it) or loses them, because every record that
            # carried it predates the fixture change (its reason becomes
            # `no reference`, and what is owed is a re-record, not a promotion).
            if lane not in stale:
                wrong_reason.append(
                    f"{lane}: NOT stale. "
                    + (f"the table carries cells at the current revision {revisions[lane]!r}, so the hold no "
                       f"longer applies: promote it once a CPU-only `verify --all` has been watched to read "
                       f"IDENTICAL for it, as the promotion rule requires"
                       if lane in with_cells else
                       f"the table carries NO cell for it at all, so its reason is 'no reference' and what is "
                       f"owed is a record at the current fixture revision {revisions[lane]!r}"))
        elif why == "no reference":
            if lane in with_cells:
                wrong_reason.append(f"{lane}: the shipped table DOES carry cells for it; let it in")
        elif why == "unwatched":
            # THE ONE REASON A REGENERATION CANNOT CLEAR BY ITSELF
            # (lane/expose-stepfull, 2026-09-16). It says the static
            # conditions are all met and only the watched CPU-only run is
            # owed, so it is checked against exactly that: the table must
            # carry cells for the lane AT THE CURRENT REVISION. A lane that
            # loses its cells, or whose fixture moves again, cannot hide
            # here; it falls back to `no reference` or `stale reference`.
            if lane not in with_cells:
                wrong_reason.append(f"{lane}: held back as 'unwatched', but the shipped table carries "
                                    "NO cell for it, so its reason is 'no reference' and what is owed "
                                    "is a record, not a run")
            elif lane in stale:
                wrong_reason.append(f"{lane}: held back as 'unwatched', but its fixture has moved past "
                                    f"the shipped reference ({revisions.get(lane)!r}), so its reason is "
                                    "'stale reference' and a run would prove nothing")
        elif why == "qualification pending":
            assert lane in with_cells and lane not in stale, lane
            assert len(_reference_classes(table, lane)) >= 2, lane
        elif why == "one column":
            # THE REFERENCE HAS ONE WITNESS (lane/reference-regen, 2026-09-17).
            # The lanes whose fixture or arithmetic moved lost every cell they
            # had from the three GPU columns, and the CPU column that lane
            # recorded gave them cells again. Cells are not the whole
            # condition: a reference ONE device class has ever produced is a
            # number nothing has reproduced, and a user who disagreed with it
            # could not tell their machine apart from our single column. That
            # is the bar `PUBLIC_REFERENCE_CANDIDATES` already holds svc-poly
            # to; this states it as a condition the shipped table answers
            # instead of as prose in a comment.
            #
            # It is checked BOTH ways on purpose. A lane that loses its cells
            # cannot hide here (its reason is `no reference`), a lane whose
            # fixture moves again cannot either (`stale reference`), and a
            # lane that GAINS a second class must stop hiding here and move to
            # `unwatched`, so the next release record's columns force the list
            # to shrink rather than letting it sit.
            if lane not in with_cells:
                wrong_reason.append(f"{lane}: held back as 'one column', but the shipped table "
                                    "carries NO cell for it, so its reason is 'no reference' and "
                                    "what is owed is a record, not a second column")
            elif lane in stale:
                wrong_reason.append(f"{lane}: held back as 'one column', but its fixture has moved "
                                    f"past the shipped reference ({revisions.get(lane)!r}), so its "
                                    "reason is 'stale reference' and a second column of the old "
                                    "bytes would prove nothing")
            else:
                classes = _reference_classes(table, lane)
                if len(classes) > 1:
                    wrong_reason.append(
                        f"{lane}: held back as 'one column', but the shipped table's cells for it "
                        f"rest on {len(classes)} device classes ({', '.join(sorted(classes))}), so "
                        "the hold no longer applies: its reason is 'unwatched' until a CPU-only "
                        "`verify --all` has been watched to read IDENTICAL for it")
        elif why == "own record":
            assert lane in host_surface.TRAINING_FIX_LANES, f"{lane}: not a TRAINING_FIX_LANES lane"
        elif why.startswith("measured"):
            assert lane in with_cells, f"{lane}: held back on a measurement but the table has no cell to measure"
        else:
            raise AssertionError(f"{lane}: {why!r} is not a reason this test knows how to check")

    assert wrong_reason == [], (
        "these lanes are held back for a reason the shipped table no longer supports:\n  "
        + "\n  ".join(wrong_reason))

    # A prefix-excluded lane is never in the public set, so it cannot become
    # public carrying a stale reference and it must not be required in
    # PUBLIC_PENDING_LANES, which only admits COVERED lanes (the loop above).
    # The case first arose with par-graph-umap (lane/umap-batch-fix,
    # 2026-09-16), the first `par-` lane to get a LANE_REVISIONS entry: the
    # assertion demanded an entry that the same test's own loop would then
    # reject.
    excluded = {lane for lane in revisions
                if lane.startswith(host_surface.PUBLIC_EXCLUDED_PREFIXES)}
    moved = sorted(stale - set(host_surface.PUBLIC_PENDING_LANES)
                   - host_only - excluded)
    assert moved == [], (
        f"these lanes moved past the shipped reference and they are still public: {moved}. "
        "Add them to PUBLIC_PENDING_LANES as 'stale reference' until the release regenerates the table"
    )


def _fixture_floors():
    """`tools/fixture_floors.py` loaded by path, like the lane readers above,
    so this test needs no numpy and no bindings."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fixture_floors", ROOT / "tools" / "fixture_floors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_fixture_is_shrunk_below_its_declared_floor():
    """A FLOOR IN PROSE IS NOT A FLOOR (lane/shrink-floors, 2026-09-16).

    docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md section 1f is
    titled "LEFT BIG: samba-untied-dropout-accum" and says three steps is that
    lane's floor because step 3 is the first that evaluates the cosine arm of
    its schedule. The next lane cut it to one step anyway, FIXTURE_SHRINK_SCOPE
    carried forward only the rows half of the reasoning, and a third document
    then recorded as fact that the third step was kept. Nobody lied; the floor
    simply had no mechanism. It has one now, on the lane, in `@floor(...)`,
    and this is the gate half of it. The checker also runs in light-checks,
    which is the workflow that starts by itself."""
    mod = _fixture_floors()
    bad = mod.check(path=str(ROOT / "tools" / "identity_break.py"))
    assert bad == [], "fixture floor violations:\n  " + "\n  ".join(bad)


def test_the_fixture_floor_check_refuses_a_violating_shrink(capsys):
    """A CHECK THAT HAS NOT BEEN SEEN TO REFUSE ANYTHING is the prose floor it
    replaces. `--self-test` mutates the real harness source five ways (a cut
    below a floor, a deleted floor, a site that stops reading the floored
    local, an untraceable reason, and the exemption list used on a lane that
    has a floorable dimension) and requires each one to be REFUSED by name."""
    mod = _fixture_floors()
    assert mod.self_test(path=str(ROOT / "tools" / "identity_break.py")) is True, capsys.readouterr().out


def _lane_accounting():
    """`tools/lane_accounting.py` loaded by path, like the readers above, so
    this test needs no built binding."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("lane_accounting", ROOT / "tools" / "lane_accounting.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_lane_is_either_in_the_shipped_table_or_declared_pending():
    """NEITHER IS NOT A STATE A LANE MAY BE IN (lane/new-lane-reference-promotion,
    2026-09-20).

    Two mechanisms account for a lane. The SHIPPED TABLE gives it a number an
    installed `verify --all` can check, and `PUBLIC_PENDING_LANES` says, with
    the reason written down, that it has none. Every test above checks one or
    the other. None of them asks whether a lane is in EITHER, and on
    2026-09-20 three were in neither: `par-forecast-arima`,
    `par-forecast-holtwinters` and `par-ivf`, all added since v0.8.8, all SEEN
    on a vendor class in a committed column, none of it in the artifact the
    wheel ships.

    Neither list was wrong on its own terms, which is exactly how this
    happens. `PUBLIC_PENDING_LANES` only ever admits lanes that could become
    PUBLIC, and `PUBLIC_EXCLUDED_PREFIXES` excludes every `par-` driver from
    the public set -- the test above even states that a prefix-excluded lane
    "must not be required in PUBLIC_PENDING_LANES", and it is right. So one
    mechanism refused the lanes for a good reason and handed them to the
    other, which had not admitted a record for them yet. Each refusal was
    correct and the lanes were still invisible to both.

    This is the check that owns the gap between them. It is deliberately the
    weakest possible statement -- accounted for AT ALL, not accounted for
    WELL -- because the strong statements already have tests and a lane they
    never see cannot fail one."""
    mod = _lane_accounting()
    bad = mod.check()
    assert bad == [], "lanes that no mechanism accounts for:\n  " + "\n  ".join(bad)


def test_the_lane_accounting_check_refuses_every_way_the_gap_comes_back(capsys):
    """A CHECK THAT HAS NOT BEEN SEEN TO REFUSE ANYTHING is the counting
    nobody did. `--self-test` builds eight worlds from the real tree -- a
    lane whose table cells vanish, a deleted pending declaration, a new lane
    registered and nothing else, an UNCOVERED lane losing its cells (which
    must not be told to use a list that cannot take it), a lane reader that
    under-reads, a reader that disagrees with the harness's own LANES, a
    pending reason that outlived its lane, and a blank reason -- and requires
    each to be REFUSED BY NAME.

    It also asserts the clean direction first: a lane with no table cells
    that IS declared pending reads clean, which is the claim that makes
    declaring a real alternative to promoting rather than a second-class
    one."""
    mod = _lane_accounting()
    assert mod.self_test(verbose=True) is True, capsys.readouterr().out
