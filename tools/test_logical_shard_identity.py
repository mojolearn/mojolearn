# SPDX-License-Identifier: Apache-2.0
"""tools/logical_shard_identity.py without a device: the fixed list, the fail-closed verdicts, the summary the
release judges, and the unconditional wiring in the box job. (The shard boundaries, per-shard seeds and fold order
live in compiled Mojo dispatchers that no binding exposes; they are checked on the box by the tool itself.)"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import logical_shard_identity as LS  # noqa: E402


def part(name, sha):
    return dict(name=name, shape=[2], dtype="<f4", sha256=sha * 64)


def doc(name, arms, ranks=None):
    """A worker document; each arm spread over `ranks[c]` ranks (default: all of its c devices)."""
    ranks = ranks or {}
    return dict(lane=name, error=None, arms={str(c): dict(parts=p, error=None, ranks=ranks.get(c, c))
                                             for c, p in arms.items()})


def test_the_list_is_short_fixed_and_well_formed():
    assert LS.lane_errors() == []
    assert 6 <= len(LS.LANES) <= 12
    assert LS.COUNTS[0] == 1 and len(LS.COUNTS) >= 2


def test_a_malformed_list_is_refused():
    row = LS.LANES[0]
    assert any("duplicate" in e for e in LS.lane_errors(LS.LANES[:6] + (row,)))
    assert any("not 6..12" in e for e in LS.lane_errors(LS.LANES[:3]))
    bad = (row[0] + "-x", "other.mojo", "no-such-recipe") + row[3:]
    assert any("unknown recipe" in e for e in LS.lane_errors(LS.LANES[:6] + (bad,)))


def test_devices_repeat_the_one_card():
    assert LS.devices_for(1) == (0,)
    assert LS.devices_for(3) == (0, 0, 0)
    assert LS.devices_for(2, 1) == (1, 1)


def test_equal_arms_match_and_a_changed_arm_differs():
    same = {c: [part("a", "1")] for c in LS.COUNTS}
    assert LS.lane_verdict("x", doc("x", same))[0] == "MATCH"
    changed = dict(same)
    changed[LS.COUNTS[-1]] = [part("a", "2")]
    verdict, reason, digests = LS.lane_verdict("x", doc("x", changed))
    assert verdict == "DIFFER" and "n%d" % LS.COUNTS[-1] in reason
    line = LS.shardcheck_line("x", verdict, reason, digests)
    assert line.startswith("SHARDCHECK x single=") and " n2=" in line and " DIFFER " in line


def test_an_arm_that_did_not_spread_its_work_fails():
    same = {c: [part("a", "1")] for c in LS.COUNTS}
    multi = LS.COUNTS[1]
    verdict, reason, _ = LS.lane_verdict("x", doc("x", same, {multi: 1}))
    assert verdict == "FAIL" and "n%d spread its work over 1 rank" % multi in reason
    unknown = doc("x", same)
    unknown["arms"][str(multi)].pop("ranks")
    verdict, reason, _ = LS.lane_verdict("x", unknown)
    assert verdict == "FAIL" and "no rank record" in reason
    # Two of three logical devices receiving work is a real split: MATCH.
    assert LS.lane_verdict("x", doc("x", same, {c: min(c, 2) for c in LS.COUNTS}))[0] == "MATCH"
    # A bit difference outranks the spread verdict.
    changed = dict(same)
    changed[LS.COUNTS[-1]] = [part("a", "2")]
    assert LS.lane_verdict("x", doc("x", changed, {multi: 1}))[0] == "DIFFER"


def test_ranks_are_read_from_every_loaded_binding_and_cleared():
    class Binding:
        def __init__(self, ranks):
            self.ranks = ranks

        def device_plan_log(self):
            got, self.ranks = self.ranks, 0
            return [], got

    class Old:
        def device_plan_log(self):
            return []

    class Broken:
        __name__ = "broken"

        def device_plan_log(self):
            raise RuntimeError("binding lost\nits table")

    a, b = Binding(3), Binding(1)
    assert LS.ranks_take({"a": a, "b": b, "again": a, "old": Old(), "plain": object()}) == 3
    assert LS.ranks_take({"a": a, "b": b}) == 0
    assert LS.ranks_take({"old": Old()}) is None
    # a binding whose device_plan_log raises is skipped and named, never an exception that loses the arm
    errors = []
    assert LS.ranks_take({"broken": Broken(), "a": Binding(2)}, errors) == 2
    assert len(errors) == 1 and "broken.device_plan_log: RuntimeError: binding lost its table" == errors[0]
    assert LS.ranks_take({"broken": Broken()}) is None


def test_a_broken_rank_record_names_the_binding_in_the_fail_reason():
    same = {c: [part("a", "1")] for c in LS.COUNTS}
    multi = LS.COUNTS[1]
    broken = doc("x", same)
    broken["arms"][str(multi)].pop("ranks")
    broken["arms"][str(multi)]["rank_errors"] = ["_mojolearn_svm.device_plan_log: RuntimeError: boom"]
    verdict, reason, _ = LS.lane_verdict("x", broken)
    assert verdict == "FAIL" and "no rank record" in reason and "boom" in reason


def test_the_gram_lane_is_a_route_that_reaches_the_gram_plan():
    # Ridge / LinearRegression take linear_gram_fit (no DevicePlan) in IDENTICAL: never on the list.
    rows = {r[0]: r for r in LS.LANES}
    assert "ridge-gram" not in rows and all(r[2] != "ridge" for r in LS.LANES)
    name, family, recipe, _fixture, n_rows, features, _seed = rows["tsvd-gram"]
    assert family == "core/gram_multi_gpu.mojo" and recipe == "tsvd" and LS.RECIPES[recipe] is LS._tsvd
    # the split-K arm needs features <= 128; three live row chunks need rows >= 3 * 128
    assert features <= 128 and n_rows >= 3 * 128


def test_no_lane_names_a_dispatcher_no_identical_call_reaches():
    # every GLM takes the QN tiled gradient in IDENTICAL (glm/impl/qn/glm_base.mojo), never gradient_columns
    assert all(r[1] != "glm/impl/qn/multi_gpu.mojo" and r[2] != "logistic" for r in LS.LANES)
    name, family, recipe, _fixture, n_rows, _features, _seed = {r[0]: r for r in LS.LANES}["gp-regress"]
    assert family == "gaussian_process/checks/kernels.mojo" and LS.RECIPES[recipe] is LS._gp
    assert n_rows >= max(LS.COUNTS)  # covariance rows: active = min(count, rows)


def test_everything_missing_fails_closed():
    assert LS.lane_verdict("x", None)[0] == "FAIL"
    assert LS.lane_verdict("x", dict(lane="x", error="ImportError: no wheel"))[:2] == ("FAIL", "ImportError: no wheel")
    no_arm = doc("x", {1: [part("a", "1")]})
    assert LS.lane_verdict("x", no_arm)[0] == "FAIL"
    empty = doc("x", {c: [] for c in LS.COUNTS})
    verdict, reason, _ = LS.lane_verdict("x", empty)
    assert verdict == "FAIL" and "no digest" in reason
    raised = doc("x", {c: [part("a", "1")] for c in LS.COUNTS})
    raised["arms"]["2"]["error"] = "ValueError: devices must be distinct\nnonnegative"
    verdict, reason, _ = LS.lane_verdict("x", raised)
    assert verdict == "FAIL" and "\n" not in reason


def test_the_release_summary_requires_every_lane_on_the_list_to_match():
    lanes = {n: dict(verdict="MATCH", reason="") for n in LS.lane_names()}
    good = dict(schema=LS.SCHEMA, ok=True, counts=list(LS.COUNTS), lanes=lanes)
    assert LS.summary_errors(good) == []
    assert LS.summary_errors(None) == ["no logical-shard summary"]
    short = dict(good, lanes={n: r for n, r in list(lanes.items())[1:]})
    assert LS.summary_errors(short)
    failed = dict(good, lanes=dict(lanes, **{LS.lane_names()[0]: dict(verdict="FAIL", reason="timed out")}))
    assert any("FAIL timed out" in e for e in LS.summary_errors(failed))
    assert LS.summary_errors(json.loads(json.dumps(dict(good, counts=[1, 2]))))


def test_the_box_job_runs_it_unconditionally_and_it_decides_the_verdict():
    job = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()
    lines = job.splitlines()
    at = lines.index("PHASE=shardcheck")
    # top level: not inside the identity phase's `if [ -n "${GATE_IDENTITY_COLUMNS:-}" ]` block (or any other)
    depth = 0
    for line in lines[:at]:
        stripped = line.strip()
        if stripped.startswith("if ") and stripped.endswith("then"):
            depth += 1
        elif stripped == "fi":
            depth -= 1
    assert depth == 0
    assert "tools/logical_shard_identity.py" in job and '[ "$grc" != 0 ] || grc=$src' in job
    # its record comes home with the job's reports
    assert '"shardcheck",' in (ROOT / "tools" / "ops" / "release_gate.py").read_text()


def test_the_box_runs_the_macs_copy_of_the_tool_right_after_the_overlay():
    job = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()
    lines = job.splitlines()
    overlay, shard, gate = (lines.index("PHASE=" + p) for p in ("overlay", "shardcheck", "gate"))
    assert overlay < shard < gate
    # the tool is a gate runner overlay file: the box runs the same LANES the Mac judges with
    import ast
    tree = ast.parse((ROOT / "tools" / "ops" / "release_gate.py").read_text())
    overlay = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(getattr(t, "id", None) == "GATE_OVERLAY" for t in node.targets))
    assert "tools/logical_shard_identity.py" in overlay


def test_the_shard_venv_always_gets_numpy():
    job = (ROOT / "tools" / "ops" / "release_gate_job.sh").read_text()
    block = job[job.index("\nPHASE=shardcheck"):job.index("\nPHASE=gate")]
    install = block.index("-m pip install -q --no-cache-dir 'numpy>=1.26.4'")
    # after the venv is chosen (identity venv or the fallback venv), before the tool runs
    assert block.index('shpy="$GATE_WORK/shard-venv/bin/python"') < install < block.index("logical_shard_identity.py --python")


def test_the_worker_runs_the_shipped_defaults(monkeypatch):
    for k, v in (("MOJOLEARN_GBDT_SHARD_ONE_DEVICE", "1"), ("MOJOLEARN_VENDOR", "cpu"),
                 ("MOJOLEARN_NUMERIC_MODE", "fast"), ("MOJOLEARN_KMEANS_DEVICE_COUNT", "1"),
                 ("PYTHONPATH", "/src"), ("CUDA_VISIBLE_DEVICES", "0")):
        monkeypatch.setenv(k, v)
    env = LS.worker_env(0)
    assert {k for k in env if k.startswith("MOJOLEARN_")} == {LS.LOGICAL_ENV, "MOJOLEARN_NUMERIC_MODE"}
    assert env[LS.LOGICAL_ENV] == "1" and env["MOJOLEARN_NUMERIC_MODE"] == "identical"
    assert "PYTHONPATH" not in env and env["CUDA_VISIBLE_DEVICES"] == "0"
