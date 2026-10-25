"""Real-shaped producer records with inert artifact bytes; no build or import."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import release_compile_evidence as compile_evidence
import release_requirement_receipts as receipts
import release_requirements as requirements


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return compile_evidence.reference(path)


def fixture(root, kind="binding"):
    source = "a" * 40
    set_root = root / "build/sets/cuda/sm_89"
    binary = set_root / "identical/_mojolearn_native.so"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"inert compile-test artifact, never loaded")
    binary_sha = compile_evidence.reference(binary)["sha256"]
    files = {"_mojolearn_native.so": ["_mojolearn_test"]}
    index = write(binary.parent / "_mojolearn_merged.json", {"files": files})
    toolchain = dict(packages=["mojo-1.0.0-release"], platform="linux-64", pixi_lock_sha256="b" * 64)
    binding = dict(target="linux-64", vendor="cuda", arch="sm_89", tier="identical",
                   name="_mojolearn_test", toolchain=toolchain, flags={"numeric_mode": "identical"})
    key = compile_evidence.binding_key(binding)
    runtime = dict(target="linux-64", toolchain=toolchain, builders={"stage.py": "f" * 64})
    inputs = dict(unit=dict(kind="compile", vendor="nvidia", release_modes=["identical"], build_flags={},
                           compiler_sha256="c" * 64),
                  binding=binding)
    coverage = [compile_evidence.BASE_COVERAGE, key]
    inventory = [["bindings/test.mojo", "d" * 64]]
    adapters = {}
    manifest = {"extensions": [binary.name]}
    if kind == "runtime":
        inputs.pop("binding"); inputs["runtime"] = runtime
        coverage = [compile_evidence.BASE_COVERAGE, "native-runtime-libraries"]
        lib = set_root / ".libs/libMojolearnMath.so"
        lib.parent.mkdir(); lib.write_bytes(b"inert runtime library")
        manifest.update(staged_libs=[dict(name=lib.name, sha256=compile_evidence.reference(lib)["sha256"])],
                        input_edges=[dict(needs=lib.name, **{"class": "staged"})])
    elif kind == "adapters":
        inputs.pop("binding")
        inputs.update(adapter_sources=dict(inventory), environment={k: binding[k] for k in
                      ("target", "vendor", "arch", "toolchain", "flags")})
        inputs["environment"]["builders"] = {"build.sh": "e" * 64}
        coverage = [compile_evidence.BASE_COVERAGE, "native-adapter-library", "torch-registration", "max-operation-package"]
        directory = set_root / "_adapters"; directory.mkdir()
        roles = dict(native="libmojolearn_adapter.so", torch="mojolearn_torch.so", max="mojolearn_ops.mojoc")
        members, rows = {}, {}
        for role, name in roles.items():
            path = directory / name; path.write_bytes(("inert " + role).encode())
            members[name] = compile_evidence.reference(path)["sha256"]
            rows[role] = dict(name=name, sha256=members[name], bytes=path.stat().st_size)
        write(directory / "adapters.json", dict(schema="mojolearn.adapters.v1", abi=1, vendor="cuda", arch="sm_89",
              platform="linux", numeric_mode="identical", source_commit=source, sources=dict(inventory), files=rows,
              torch=dict(series="2.13", backend="cuda", wheel_sha256="e" * 64, cxx11_abi=1),
              max=dict(version="26.5.0"), commands={"native": ["mojo", "build"]}))
        members["adapters.json"] = compile_evidence.reference(directory / "adapters.json")["sha256"]
        adapters["mojolearn/cuda/sm_89/_adapters"] = dict(files=members, manifest_sha256=members["adapters.json"])
    name = "compile-nvidia/" + kind
    wanted = dict(kind="compile", coverage=coverage, source_revision=source, inputs=inputs,
                  fingerprint=compile_evidence.digest(inputs))
    capture = dict(schema=compile_evidence.CAPTURE_SCHEMA, phase="before-build", source_commit=source,
        source_inventory_digest=compile_evidence.digest(inventory), vendor="cuda", arch="sm_89",
        binding_identities={key: binding}, runtime_identity=runtime, adapter_builders={"build.sh": "e" * 64},
        toolchain=toolchain, installed_packages=toolchain["packages"], compiler_sha256="c" * 64, build_flags={})
    capture_ref = write(root / "build/compile-inputs.json", capture)
    launch_ref = write(root / "leg.provenance.json", dict(source_commit=source, launched_at="2026-10-25T01:00:00Z",
        vendor="cuda", arch="sm_89", compile_requirements=dict(schema=compile_evidence.LAUNCH_SCHEMA,
        runs={name: dict(run_id="run-one", fingerprint=wanted["fingerprint"], source_revision=source)})))
    manifest_ref = write(set_root / "manifest.json", manifest)
    prefix = "mojolearn/cuda/sm_89/"
    proof = dict(schema=compile_evidence.BUILD_SCHEMA, action="build", complete=True, build_exit=0,
        source_commit=source, source_inventory=inventory,
        source_sha256=hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(),
        release_modes=["identical"], compile_inputs_sha256=capture_ref["sha256"], binding_layout="merged",
        extensions={prefix + "identical/" + binary.name: binary_sha}, host_extension={}, adapters=adapters,
        merged_indexes={prefix + "identical/_mojolearn_merged.json": dict(sha256=index["sha256"], files=files)},
        set_manifests={prefix + "manifest.json": manifest_ref["sha256"]})
    proof_ref = write(root / "build/build-provenance.json", proof)
    exit_file = root / "leg.exit"; exit_file.write_text("0\n")
    report_path = root / "receipt/compile-report.json"
    report = dict(schema=compile_evidence.REPORT_SCHEMA, producer=compile_evidence.TOOL, build_report=proof_ref,
        capture_report=capture_ref, launch_report=launch_ref, set_manifest=manifest_ref,
        exit_report=compile_evidence.reference(exit_file), set_root=str(set_root))
    write(report_path, report)
    stamp = dict(schema=receipts.PROVENANCE_SCHEMA, requirement_id=name, kind="compile", source_revision=source,
                 input_fingerprint=wanted["fingerprint"], report_sha256=compile_evidence.reference(report_path)["sha256"],
                 run_id="run-one")
    write(report_path.parent / "producer-inputs.json", stamp)
    return name, wanted, report_path, stamp


def test_actual_linux_report_derives_coverage_and_revalidates_bytes(tmp_path):
    name, wanted, report, provenance = fixture(tmp_path)
    receipt = compile_evidence.producer_payload(name, wanted, report, provenance)
    assert receipts.validate_receipt(name, wanted, receipt) == []
    assert {row["id"] for row in receipt["coverage"]} == set(wanted["coverage"])
    assert len(receipt["evidence"]) >= 8
    artifact = next(row for row in receipt["evidence"] if row["path"].endswith(".so"))
    Path(artifact["path"]).write_bytes(b"corrupt")
    assert receipts.validate_receipt(name, wanted, receipt)


@pytest.mark.parametrize("kind", ["runtime", "adapters"])
def test_c_and_cpp_units_wait_for_their_actual_compiler_byte_identity(tmp_path, kind):
    name, wanted, report, provenance = fixture(tmp_path, kind)
    assert not compile_evidence.supports_requirement(wanted)
    with pytest.raises(ValueError, match="unsupported compile target/configuration"):
        compile_evidence.producer_payload(name, wanted, report, provenance)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    status = ledger.sync(dict(schema=requirements.SCHEMA, version="0.8.57", requirements={name: wanted},
                             source_revision=wanted["source_revision"], source_root=str(tmp_path)))
    assert status[name]["action"] == "complete-provenance"
    with pytest.raises(ValueError, match="provenance"):
        ledger.begin(name)


def test_vendor_host_bindings_have_one_real_leg_owner(tmp_path):
    _, wanted, _, _ = fixture(tmp_path)
    binding = wanted["inputs"]["binding"]
    binding.update(vendor="host", arch="", tier="host")
    wanted["coverage"][1] = compile_evidence.binding_key(binding)
    assert compile_evidence.belongs_to_leg(wanted, "cuda", "sm_89")
    assert not compile_evidence.belongs_to_leg(wanted, "cuda", "sm_80")
    assert not compile_evidence.belongs_to_leg(wanted, "hip", "gfx942")
    wanted["inputs"]["unit"]["vendor"] = "amd"
    assert compile_evidence.belongs_to_leg(wanted, "hip", "gfx942")


@pytest.mark.parametrize("damage", ["development", "missing-capture", "wrong-source", "wrong-flags", "compiler", "compiler-bytes",
    "source-inventory", "binding-identity", "missing-member", "duplicate-owner", "failed", "wrong-launch", "manifest"])
def test_stale_incomplete_or_unbound_producers_refuse(tmp_path, damage):
    name, wanted, path, stamp = fixture(tmp_path)
    report = json.loads(path.read_text())
    target = "capture_report" if damage in ("wrong-flags", "compiler", "compiler-bytes", "binding-identity") else "build_report"
    if damage == "wrong-launch": target = "launch_report"
    doc_path = Path(report[target]["path"]); doc = json.loads(doc_path.read_text())
    if damage == "development": doc["schema"] = "mojolearn.development-group.v1"
    elif damage == "missing-capture": doc.pop("compile_inputs_sha256")
    elif damage == "wrong-source": doc["source_commit"] = "f" * 40
    elif damage == "wrong-flags": doc["build_flags"] = {"MOJOLEARN_BUILD_DEFINES": "NEW_KERNEL=1"}
    elif damage == "compiler": doc["installed_packages"] = []
    elif damage == "compiler-bytes": doc["compiler_sha256"] = "f" * 64
    elif damage == "source-inventory": doc["source_inventory"][0][1] = "a" * 64
    elif damage == "binding-identity": next(iter(doc["binding_identities"].values()))["flags"] = {}
    elif damage == "missing-member": next(iter(doc["merged_indexes"].values()))["files"] = {}
    elif damage == "duplicate-owner": next(iter(doc["merged_indexes"].values()))["files"]["another.so"] = ["_mojolearn_test"]
    elif damage == "failed": doc["build_exit"] = 1
    elif damage == "wrong-launch": doc["compile_requirements"]["runs"][name]["run_id"] = "some-other-run"
    elif damage == "manifest": doc["set_manifests"] = {}
    report[target] = write(doc_path, doc)
    # Refresh only ordinary file references: parsing must still reject the
    # altered actual facts, rather than relying solely on a stale wrapper hash.
    if target == "capture_report":
        proof = json.loads(Path(report["build_report"]["path"]).read_text())
        proof["compile_inputs_sha256"] = report[target]["sha256"]
        report["build_report"] = write(Path(report["build_report"]["path"]), proof)
    write(path, report); stamp["report_sha256"] = compile_evidence.reference(path)["sha256"]
    write(path.parent / "producer-inputs.json", stamp)
    with pytest.raises((ValueError, KeyError)):
        compile_evidence.producer_payload(name, wanted, path, stamp)


def test_undeclared_coverage_apple_and_variants_do_not_enable_execution(tmp_path):
    name, wanted, report, stamp = fixture(tmp_path)
    assert requirements.producer_admission_available(wanted)
    for mutation in ("apple", "extra-coverage", "variants"):
        other = copy.deepcopy(wanted)
        if mutation == "apple": other["inputs"]["binding"]["target"] = "osx-arm64"
        elif mutation == "extra-coverage": other["coverage"].append("unproved-switch-off")
        else: other["inputs"]["configuration"] = {"compile_variants": ["off", "on"]}
        assert not requirements.producer_admission_available(other)


def test_source_only_log_cannot_be_a_compile_producer(tmp_path):
    name, wanted, report, stamp = fixture(tmp_path)
    report.write_text("No compilation occurred.\n")
    with pytest.raises(ValueError):
        compile_evidence.producer_payload(name, wanted, report, stamp)


def test_active_producer_cannot_be_begun_again(tmp_path):
    name, wanted, report, stamp = fixture(tmp_path)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(dict(schema=requirements.SCHEMA, requirements={name: wanted}, version="0.8.57",
                     source_revision=wanted["source_revision"], source_root=str(tmp_path)))
    run = ledger.begin(name)
    # Another active producer cannot be duplicated by a batch.
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="duplicate work"):
        ledger.begin_many([name])
    assert ledger.path.read_bytes() == before
    assert len(ledger.read()["runs"]) == 1 and run


def test_batch_later_refusal_keeps_no_partial_new_runs(tmp_path):
    name, wanted, report, stamp = fixture(tmp_path)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    other = copy.deepcopy(wanted)
    other["missing_inputs"] = ["missing original toolchain proof"]
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(dict(schema=requirements.SCHEMA, requirements={name: wanted, "blocked": other}, version="0.8.57",
                     source_revision=wanted["source_revision"], source_root=str(tmp_path)))
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="provenance"):
        ledger.begin_many([name, "blocked"])
    assert ledger.path.read_bytes() == before
    assert ledger.read()["runs"] == []


def test_partial_missing_compiler_inputs_do_not_enable_new_work(tmp_path):
    _, wanted, _, _ = fixture(tmp_path)
    wanted["missing_inputs"] = ["compiler unavailable"]
    assert not requirements.producer_admission_available(wanted)


def test_capture_allows_only_explicit_build_inputs_without_credentials():
    secret = "fixture-secret-never-retained"
    env = {"MOJOLEARN_BUILD_DEFINES": "FEATURE=0", "AWS_SECRET_ACCESS_KEY": secret,
           "PIP_INDEX_URL": "https://" + secret + "@example.invalid/", "MOJOLEARN_PRIVATE_KEY": secret,
           "MOJOLEARN_REQUIREMENTS_LEDGER": "/some/evidence/location"}
    assert compile_evidence.safe_build_flags(env) == {"MOJOLEARN_BUILD_DEFINES": "FEATURE=0"}
    with pytest.raises(ValueError) as caught:
        compile_evidence.safe_build_flags({"MOJOLEARN_UNDECLARED_INPUT": secret})
    assert secret not in str(caught.value)


@pytest.mark.parametrize("environment", [None, "default"])
def test_capture_reads_the_environment_build_sets_will_use(tmp_path, monkeypatch, environment):
    import release_reuse as reuse
    if environment is None:
        monkeypatch.delenv("MOJOLEARN_BUILD_PIXI_ENV", raising=False)
    else:
        monkeypatch.setenv("MOJOLEARN_BUILD_PIXI_ENV", environment)
    for name in ("default", "gbmbench"):
        prefix = tmp_path / ".pixi/envs" / name
        (prefix / "bin").mkdir(parents=True); (prefix / "conda-meta").mkdir()
        (prefix / "bin/mojo").write_bytes((name + " inert compiler bytes, not executable").encode())
        (prefix / "conda-meta/mojo-pin.json").write_text("{}\n")
    output = tmp_path / "build/compile-inputs.json"
    write(output.parent / "source-inventory.json", [])
    monkeypatch.setattr(reuse, "ROOT", tmp_path)
    monkeypatch.setattr(reuse, "tree_facts", lambda _: {"toolchain": {reuse.LINUX: {"packages": ["mojo-pin"]}}})
    monkeypatch.setattr(reuse, "bindings", lambda *a: [])
    monkeypatch.setattr(reuse, "runtime_identity", lambda _: {})
    monkeypatch.setattr(reuse, "LINUX_BUILDERS", [])
    monkeypatch.setattr(compile_evidence, "safe_build_flags", lambda _: {})
    record = compile_evidence.capture(tmp_path, output, "a" * 40, "cuda", "sm_89")
    expected = tmp_path / ".pixi/envs" / (environment or "gbmbench") / "bin/mojo"
    assert record["compiler_sha256"] == compile_evidence.reference(expected)["sha256"]


def test_real_compile_receipt_reuses_on_a_new_source_commit(tmp_path):
    name, wanted, report_path, stamp = fixture(tmp_path)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    manifest = dict(schema=requirements.SCHEMA, requirements={name: wanted}, version="0.8.57",
                    source_revision=wanted["source_revision"], source_root=str(tmp_path))
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(manifest)
    run_id = ledger.begin(name)
    report = json.loads(report_path.read_text())
    launch_path = Path(report["launch_report"]["path"])
    launch = json.loads(launch_path.read_text())
    launch["compile_requirements"]["runs"][name]["run_id"] = run_id
    report["launch_report"] = write(launch_path, launch)
    write(report_path, report)
    stamp.update(run_id=run_id, report_sha256=compile_evidence.reference(report_path)["sha256"])
    write(report_path.parent / "producer-inputs.json", stamp)
    receipt = compile_evidence.producer_payload(name, wanted, report_path, stamp)
    ledger.record(run_id, "passed", [report_path], "actual retained compiler proof", 0,
                  receipt=receipt, **receipts.coverage_summary(receipt))
    later = copy.deepcopy(manifest)
    later["source_revision"] = later["requirements"][name]["source_revision"] = "f" * 40
    assert ledger.sync(later)[name]["action"] == "reuse"
    assert ledger.begin_many([name]) == {name: None}
    assert len(ledger.read()["runs"]) == 1


def test_compile_admission_refusal_precedes_rotation_and_spawn(tmp_path):
    import release
    from types import SimpleNamespace
    leg = release.Leg("cuda-sm_89", "cuda", "sm_89", ["never", "run"], {},
                      tmp_path / "output/release-build", tmp_path / "logs", tmp_path / "output")
    leg.workdir.mkdir(); leg.out_dir.mkdir()
    leg.log.write_text("retained original failure\n")
    retained = leg.out_dir / "original.json"; retained.write_text("{}\n")
    spawned = []
    def refused(_leg):
        raise release.StepFailed("missing original compiler proof")
    ctx = SimpleNamespace(commit="a" * 40, dry=False, say=lambda _: None, begin_compile_requirements=refused,
                          spawn=lambda *args: spawned.append(args))
    with pytest.raises(release.StepFailed, match="compiler proof"):
        release.launch_detached(ctx, [leg])
    assert spawned == []
    assert leg.log.read_text() == "retained original failure\n" and retained.exists()


@pytest.mark.parametrize("variants", [[], ["off", "on"]])
def test_real_dependency_expansion_preserves_variants_and_expected_compiler(tmp_path, monkeypatch, variants):
    import release_requirement_dependencies as deps
    from test_release_requirement_dependencies import snapshot, spec
    snap = snapshot()
    for row in snap["bindings"]:
        row["identity"]["tier"] = row["tier"]
    wanted_spec = spec("compile")
    wanted_spec["coverage"] = [compile_evidence.BASE_COVERAGE]
    monkeypatch.setattr(deps, "frozen_snapshot", lambda *a, **k: snap)
    monkeypatch.setattr(requirements, "source_files", lambda *a: {})
    config = dict(build_flags={}, release_modes=["identical"], compile_variants=variants,
                  dependency_context=dict(compiler_sha256={"linux-64": "c" * 64}))
    catalog = dict(requirements=[wanted_spec])
    result = requirements.requirements(tmp_path, "a" * 40, "0.8.57", catalog, config,
                                       expand_dependencies=True)
    row = next(v for k, v in result["requirements"].items() if k.endswith("/one"))
    assert row["inputs"]["unit"]["compile_variants"] == variants
    assert compile_evidence.supports_requirement(row) is (not bool(variants))
    config["dependency_context"] = {}
    missing = requirements.requirements(tmp_path, "a" * 40, "0.8.57", catalog, config,
                                        expand_dependencies=True)
    row = next(v for k, v in missing["requirements"].items() if k.endswith("/one"))
    assert "dependency_context.compiler_sha256.linux-64 is required" in row["missing_inputs"]
    assert not compile_evidence.supports_requirement(row)


def failure_controller(tmp_path):
    import release
    from types import SimpleNamespace
    name, wanted, report_path, stamp = fixture(tmp_path)
    wanted.update(scope="test", required=True, stage="pre", missing_inputs=[])
    manifest = dict(schema=requirements.SCHEMA, requirements={name: wanted}, version="0.8.57",
                    source_revision=wanted["source_revision"], source_root=str(tmp_path))
    ledger = requirements.RequirementsLedger(tmp_path / "requirements.json")
    ledger.sync(manifest); run = ledger.begin(name)
    leg = release.Leg("cuda-sm_89", "cuda", "sm_89", [], {}, tmp_path,
                      tmp_path / "logs", tmp_path / "outputs")
    leg.workdir.mkdir(); leg.log.write_text("actual compiler diagnostic\n"); leg.exit_file.write_text("1\n")
    launch = json.loads((tmp_path / "leg.provenance.json").read_text())
    launch["compile_requirements"]["runs"][name]["run_id"] = run
    launch["compile_requirements"]["selected"] = [name]
    write(leg.provenance_file, launch)
    proof_path = tmp_path / "build/build-provenance.json"
    proof = json.loads(proof_path.read_text()); proof.update(build_exit=1, complete=False); write(proof_path, proof)
    ctx = SimpleNamespace(rel=tmp_path / "release", commit=wanted["source_revision"], dry=False,
                          readiness=lambda: ledger, readiness_manifest=lambda: manifest,
                          admitted_leg=lambda _: False, reused_leg_objects=lambda: [])
    ctx.record_compile_failure = lambda l: release.Release.record_compile_failure(ctx, l)
    return ctx, ledger, leg, name, run


def test_failed_producer_records_original_run_before_wait_raises(tmp_path, monkeypatch):
    import release
    ctx, ledger, leg, name, run = failure_controller(tmp_path)
    monkeypatch.setattr(release, "linux_legs", lambda _: [leg])
    with pytest.raises(release.StepFailed, match="build leg.*failed"):
        release.Release.step_linux_wait(ctx)
    attempts = ledger.read()["attempts"]
    assert len(attempts) == 1 and attempts[0]["run_id"] == run and attempts[0]["status"] == "failed"
    assert attempts[0]["exit_code"] == 1 and attempts[0]["coverage"]["completed"] == 0
    leg.log.write_text("later rotated/replaced original log\n")
    assert requirements.evidence_errors(attempts[0]["evidence"]) == []
    assert ledger.plan()[name]["action"] == "repair"
    assert ledger.begin(name) != run


def test_successful_peer_leg_records_before_another_leg_failure_raises(tmp_path, monkeypatch):
    import release
    from types import SimpleNamespace
    ctx, ledger, failed_leg, name, run = failure_controller(tmp_path)
    success = SimpleNamespace(name="cuda-sm_80", vendor="cuda", arch="sm_80", command=[], running=lambda: False,
                              exit_code=lambda: 0, pid=lambda: None, proof_ok=lambda _: True,
                              provenance_file=tmp_path / "success.json", out_dir=tmp_path / "success")
    write(success.provenance_file, {})
    admitted = []
    ctx.record_compile_requirements = lambda leg: admitted.append(leg.name)
    monkeypatch.setattr(release, "linux_legs", lambda _: [failed_leg, success])
    monkeypatch.setattr(release, "overlay_verified", lambda *a: (True, "verified"))
    with pytest.raises(release.StepFailed, match="build leg.*failed"):
        release.Release.step_linux_wait(ctx)
    assert admitted == [success.name]
    assert ledger.read()["attempts"][0]["status"] == "failed"


@pytest.mark.parametrize("kind", ["missing-proof", "provisioning", "timeout", "signal"])
def test_interrupted_or_provisioning_only_is_not_completed_compile_failure(tmp_path, kind):
    ctx, ledger, leg, name, run = failure_controller(tmp_path)
    proof_path = tmp_path / "build/build-provenance.json"
    if kind == "missing-proof": proof_path.unlink()
    elif kind == "provisioning": (tmp_path / "build/compile-inputs.json").unlink()
    else: leg.exit_file.write_text("124\n" if kind == "timeout" else "137\n")
    ctx.record_compile_failure(leg)
    assert ledger.read()["attempts"] == []
    assert ledger.plan()[name]["action"] == "resume"


def test_record_batch_is_atomic_and_evaluates_once(tmp_path, monkeypatch):
    ctx, ledger, leg, name, run = failure_controller(tmp_path)
    row = dict(run_id=run, status="failed", paths=[leg.log], summary="compiler failed", exit_code=1,
               expected=1, completed=0)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_many([row, row])
    assert ledger.path.read_bytes() == before
    original = ledger._evaluate; evaluations = []
    def evaluated(doc):
        evaluations.append(1)
        return original(doc)
    monkeypatch.setattr(ledger, "_evaluate", evaluated)
    ledger.record_many([row])
    assert len(evaluations) == 1 and len(ledger.read()["attempts"]) == 1


@pytest.mark.parametrize("damage", [None, "empty-runs", "scope", "completed", "actual-build"])
def test_provider_retry_requires_original_nonempty_exact_active_scope(tmp_path, monkeypatch, damage):
    import release
    ctx, ledger, leg, name, run = failure_controller(tmp_path)
    leg.compile_provider_retry = True
    launch = json.loads(leg.provenance_file.read_text())
    if damage == "empty-runs": launch["compile_requirements"]["runs"] = {}
    elif damage == "scope": launch["compile_requirements"]["selected"] += ["another-selected-unit"]
    elif damage == "completed": ctx.record_compile_failure(leg)
    if damage != "actual-build":
        (tmp_path / "build/compile-inputs.json").unlink()
        (tmp_path / "build/build-provenance.json").unlink()
    write(leg.provenance_file, launch)
    monkeypatch.setattr(release, "file_at", lambda *a: compile_evidence.CAPTURE_SCHEMA)
    if damage:
        with pytest.raises(release.StepFailed, match="provider retry"):
            release.Release.begin_compile_requirements(ctx, leg)
    else:
        assert release.Release.begin_compile_requirements(ctx, leg) == launch["compile_requirements"]
    if damage == "actual-build":
        assert ledger.read()["attempts"][0]["status"] == "failed"
