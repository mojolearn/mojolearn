#!/usr/bin/env python3
"""Exercise the actual host report method without importing a GPU binding."""
import ast
import threading
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'python/mojolearn/_byte_lm_impl.py'


def method(source):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'attention_stage_report')
    env = {'state_shape': lambda state: SimpleNamespace(n_layers=3)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[node.name]


def check(fn, verbose=False):
    for n in (3, 2, 0):
        triples = [0, 2, 1, 0, 3, 0, 0, 0, 0][:3*n]
        info = [7,7,1,1,4,100,5,1,1,3,3,n] + triples + [1,1,99,0] + [0]*n + [1] + [2]*n
        fake = SimpleNamespace(_lock=threading.Lock(), _session_open=True, _native_session=object(),
                               _session_binding=SimpleNamespace(byte_lm_session_info=lambda _: info), _state={})
        r = fn(fake)
        expected = dict(stage_lists=(3,n), forward_status=[0]*n, backward_status=[2,3,0][:n],
                        attn_materialized=[True,False,False][:n], exact_tail_guard=True,
                        release_eager=True, released_eager_bytes=396, sticky_eager=False,
                        layers_prefer_eager=[False]*n, repair_masked_tail=True,
                        backward_repair_sites=[2]*n, eager_bytes=36, forward_aexp_bytes=400)
        for key, value in expected.items():
            assert r.get(key) == value, (n,key,r.get(key),value)
            if verbose: print('MATCH paired layers', n, key, value)


def main():
    source = SOURCE.read_text()
    for name, broken in (
        ('removed short-list bound', source.replace('n = min(n, int(info[10]), int(info[11]))', 'n = n')),
        ('old triple offset', source.replace('info[12 + 3 * layer + offset]', 'info[10 + 3 * layer + offset]')),
        ('wrong repair offset', source.replace('info[17 + 4 * n + layer]', 'info[16 + 4 * n + layer]')),
    ):
        assert broken != source
        try: check(method(broken))
        except (AssertionError, IndexError) as e: print('EXPECTED FAIL', name, str(e))
        else: raise AssertionError('BLIND ' + name)
    check(method(source), True)


if __name__ == '__main__': main()
