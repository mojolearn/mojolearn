# SPDX-License-Identifier: Apache-2.0
"""tools/sabotage_prebuild.py: the build -> development group mapping and the recorded sizes (no compile)."""
import json

import sabotage_prebuild as pre


def test_reuse_requires_clean_bytes_and_the_complete_patched_set(tmp_path):
    specification = {"builds": ["build"]}
    rel = "python/mojolearn/sample.so"
    patched = tmp_path / "out" / rel
    patched.parent.mkdir(parents=True)
    patched.write_bytes(b"patched")
    receipt = dict(prepared=True, clean_sha256="clean", rc={"build": 0}, changed={rel: pre.sha(patched)})
    (tmp_path / "ready").write_text(json.dumps(receipt))
    assert pre.reusable(tmp_path, specification, "clean")
    assert not pre.reusable(tmp_path, specification, "different clean build")
    patched.write_bytes(b"truncated")
    assert not pre.reusable(tmp_path, specification, "clean")
    patched.unlink()
    assert not pre.reusable(tmp_path, specification, "clean")


def test_every_plan_build_maps_to_a_development_group():
    groups = json.load(open(pre.Path(__file__).resolve().parent.parent / "bindings" / "development_groups.json"))
    # the development stems the base build records (identical-_mojolearn_dev_core, host-_mojolearn_dev_core_host)
    stems = {(mode, "_mojolearn_dev_" + name + ("_host" if mode == "host" else ""))
             for mode, table in groups["groups"].items() for name in table}
    plan = json.load(open(pre.Path(__file__).resolve().parent / "identity_lanes" / "coverage_sabotage_gpu_plan.json"))
    builds = {b for batch in plan["batches"] for b in batch["builds"]}
    assert builds and {b for b in builds if pre.dev_group(b) not in stems} == set()


def test_recorded_reads_only_a_passed_peak(tmp_path):
    (tmp_path / "peaks").mkdir()
    (tmp_path / "peaks" / "host-_mojolearn_dev_runtime_prep_host.json").write_text(
        json.dumps({"status": "passed", "peak_rss_bytes": 3 << 30, "elapsed_seconds": 685.0}))
    (tmp_path / "peaks" / "identical-_mojolearn_dev_core.json").write_text(
        json.dumps({"status": "lower-bound", "lower_bound_rss_bytes": 1 << 30}))
    assert pre.recorded(tmp_path, "build_runtime_prep_host") == (3 << 30, 685.0)
    assert pre.recorded(tmp_path, "build") == (0, 0.0)
    assert pre.recorded("", "build") == (0, 0.0)
