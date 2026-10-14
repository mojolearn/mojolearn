import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("session", Path(__file__).with_name("bench_board_session.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_worker_marker_is_exact_and_other_environment_is_not_retained(tmp_path):
    for pid, env in [(123, b"SECRET=hidden\0MOJOLEARN_BOARD_SESSION=ours\0"),
                     (456, b"MOJOLEARN_BOARD_SESSION=ours-other\0"), (789, b"UNRELATED=yes\0")]:
        p = tmp_path / str(pid)
        p.mkdir()
        (p / "environ").write_bytes(env)
    assert S.marked_processes("ours", tmp_path) == [123]


def test_long_workload_requires_provider_deadline_and_retention_reserve():
    S.validate_window(3600, now=100)
    S.validate_window(7200, provider_deadline=9000, now=100)
    for seconds, deadline in [(0, None), (14401, 20000), (3601, None),
                              (7200, 8100), (60, 100), (7200, float("nan")), (7200, float("inf"))]:
        with pytest.raises(ValueError):
            S.validate_window(seconds, deadline, now=100)
