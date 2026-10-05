"""The committed configuration witness keeps its schema and finds live collectors."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('witness', ROOT / 'tools/cuda_runtime_config_witness.py')
witness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(witness)


def fake_proc(tmp_path, argv, maps):
    proc = tmp_path / 'proc'
    for pid, (args, lines) in enumerate(zip(argv, maps), start=100):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / 'cmdline').write_bytes('\0'.join(args).encode() + b'\0')
        (proc / str(pid) / 'maps').write_text('\n'.join(lines) + '\n')
    (proc / 'self').mkdir()
    (proc / 'self' / 'maps').write_text('')
    return proc


def test_schema_and_library_filter_are_unchanged(tmp_path):
    assert witness.SCHEMA == 'mojolearn.cuda-runtime-config-witness.v1'
    cuda = tmp_path / 'libcuda.so.580.1'
    cuda.write_bytes(b'driver')
    other = tmp_path / 'libc.so.6'
    other.write_bytes(b'libc')
    line = '7f00-7f01 r-xp 00000000 08:01 1 {}'
    proc = fake_proc(tmp_path,
                     [['venv/bin/python', 'source/tools/nvidia_baseline_qualification.py', 'collect', '--out', 'results/full-baseline.json'],
                      ['venv/bin/python', 'source/tools/nvidia_baseline_qualification.py', 'check'],
                      ['sleep', '5']],
                     [[line.format(cuda), line.format(cuda), line.format(other), '7f02-7f03 rw-p 0 00:00 0'], [], []])
    rows = witness.live_collectors(proc)
    assert len(rows) == 1 and rows[0]['pid'] == 100
    assert rows[0]['libraries'] == [dict(path=str(cuda), sha256=witness.sha(cuda))]
    assert witness.ready(rows, 'results/full-baseline.json')
    assert not witness.ready(rows, 'results/full-native-reference.json')
    rows[0]['libraries'] = []
    assert not witness.ready(rows, 'results/full-baseline.json')


def test_wait_returns_only_once_the_named_collector_maps_cuda():
    live = [dict(pid=1, argv=['x', 'results/full-baseline.json'], libraries=[dict(path='/usr/lib/libcuda.so.1', sha256='0')])]
    polls = iter([[], [dict(pid=1, argv=['x', 'results/full-baseline.json'], libraries=[])], live])
    ticks = iter(range(100))
    assert witness.wait_for('results/full-baseline.json', 50, poll=lambda: next(polls),
                            sleep=lambda s: None, clock=lambda: next(ticks)) == live
    with pytest.raises(SystemExit, match='No live collector'):
        witness.wait_for('results/full-baseline.json', 2, poll=lambda: [], sleep=lambda s: None,
                         clock=lambda: next(ticks))
    assert witness.wait_for(None, 0, poll=lambda: []) == []
