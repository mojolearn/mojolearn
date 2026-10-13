import importlib.util
from pathlib import Path

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
