"""`verify --show-paths [LANE] [--from PATH]` (_verify_paths.py): reads a saved activity record, runs nothing."""
import json

import pytest

from mojolearn import __main__ as cli
from mojolearn import _verify_paths as vp


def act(launches=0, up=0, down=0, splits=0, launch_names=(), host_names=()):
    return dict(launches=launches, upload_bytes=up, upload_calls=1 if up else 0, download_bytes=down,
                download_calls=1 if down else 0, host_splits=splits, host_tasks=splits * 4, host_tasks_widest=4,
                host_entries=len(host_names), launch_names=[list(x) for x in launch_names],
                host_names=[list(x) for x in host_names])


def report(gpu=True, divergent=False):
    cells = []
    for lane, h in (("ols", "ab" * 8), ("knn", "cd" * 8)):
        for side in (("cpu", "gpu") if gpu else ("cpu",)):
            bad = divergent and lane == "knn" and side == "gpu"
            cells.append(dict(lane=lane, fixture="base", part="infer", side=side, value="ee" * 8 if bad else h,
                              state="DIVERGENT" if bad else "IDENTICAL", reference=h, family="f"))
    sides = dict(cpu=dict(lanes={"ols": act(splits=3, host_names=[("gemm_host", 2)]),
                                 "knn": act(splits=5, host_names=[("host_knn_search", 1)])}))
    if gpu:
        sides["gpu"] = dict(lanes={"ols": act(12, 4096, 64, launch_names=[("gemm_tile", 10), ("reduce", 2)]),
                                   "knn": act(7, 2048, 32, launch_names=[("knn_topk", 7)])})
    return dict(device=dict(vendor="cuda" if gpu else "cpu", device="L40S" if gpu else "cpu",
                            device_class="nvidia" if gpu else "cpu", cpu_model="EPYC"),
                sides=dict(gpu_available=gpu), cells=cells, what_ran=dict(sides=sides), lanes=["ols", "knn"],
                verdict="VERIFIED")


def run(capsys, *argv):
    args = cli.build_parser().parse_args(["verify", *argv])
    code = args.func(args)
    return code, capsys.readouterr()


def saved(tmp_path, **kw):
    path = tmp_path / "verify-all.json"
    written = vp.save(report(**kw), str(path))
    assert str(tmp_path / "verify-all.what-ran.json") in written
    return tmp_path / "verify-all.what-ran.json"


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_every_lane_side_by_side_from_the_last_run(tmp_path, capsys):
    saved(tmp_path)                                # also written to the user cache: the default --show-paths reads
    code, out = run(capsys, "--show-paths")
    assert code == 0, out.err
    row = next(line for line in out.out.splitlines() if line.startswith("ols"))
    assert "L     12" in row and "split     3" in row and "abababababababab / abababababababab" in row
    assert "2 lane(s)" in out.out


def test_one_lane_in_full(tmp_path, capsys):
    path = saved(tmp_path)
    code, out = run(capsys, "--show-paths", "ols", "--from", str(path))
    assert code == 0, out.err
    assert "gemm_host x2" in out.out and "gemm_tile x10" in out.out and "reduce x2" in out.out
    assert "to the device 4.0KB in 1 call(s)" in out.out and "same bits on both sides" in out.out


def test_a_divergent_lane_shows_the_two_hashes(tmp_path, capsys):
    path = saved(tmp_path, divergent=True)
    code, out = run(capsys, "--show-paths", "knn", "--from", str(path))
    assert code == 0
    assert "DIVERGENT: cpu cdcdcdcdcdcdcdcd (IDENTICAL) vs gpu eeeeeeeeeeeeeeee (DIVERGENT)" in out.out


def test_from_a_whole_report_of_another_machine(tmp_path, capsys):
    path = tmp_path / "theirs.json"
    path.write_text(json.dumps(report()))
    code, out = run(capsys, "--show-paths", "--from", str(path))
    assert code == 0 and "knn" in out.out


def test_a_cpu_only_record_says_the_gpu_side_was_not_run(tmp_path, capsys):
    path = saved(tmp_path, gpu=False)
    code, out = run(capsys, "--show-paths", "--from", str(path))
    assert code == 0 and "GPU side: not run here (CPU-only install)" in out.out
    code, out = run(capsys, "--show-paths", "ols", "--from", str(path))
    assert "not run here (CPU-only install)" in out.out


def test_no_record_and_unknown_lane_are_refused_in_words(tmp_path, capsys):
    code, out = run(capsys, "--show-paths")
    assert code == vp.EXIT_NO_RECORD and "verify --all" in out.err
    path = saved(tmp_path)
    code, out = run(capsys, "--show-paths", "nope", "--from", str(path))
    assert code == vp.EXIT_USAGE and "no lane 'nope'" in out.err
