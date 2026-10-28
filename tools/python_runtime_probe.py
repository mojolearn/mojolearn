#!/usr/bin/env python3
"""Empirical probe: does any Python frame run inside a public operation?

Developer audit tooling, never product runtime. For every public name the
static inventory (tools/public_runtime_inventory.py --endpoints-json)
enumerates, a child process constructs the object with
numeric_mode='identical' on the CPU column (MOJOLEARN_VENDOR=cpu), runs its
public endpoints on the ordinary verifier's tiny fixtures (the
python/mojolearn/_verify_api_* families, scikit-learn INIT_PARAMS sizes) and a
generic small-input fallback for the endpoints those families do not reach,
and records every Python frame executed with sys.setprofile armed.

Frame classes, per call:
  entry     the first product frame (python/mojolearn/, not a verifier
            fixture) entered from harness code: the public method's own frame
  inside    any Python frame executed strictly inside an entry frame,
            product or third party (numpy's Python layer, stdlib)
  callback  an inside frame entered while a native extension call of the
            enclosing frame was active: native code re-entering Python
  c_call    C functions called from entry/inside frames, by category
            (native = a mojolearn binary, numpy, builtin, other)

An endpoint whose public op ran only its entry frame plus native calls is
clean. Anything recorded as inside or callback is a finding. Frames are keyed
(file, function, first line). The report is CPU column only, small inputs,
and says nothing about bit identity, GPUs or performance.

    python3 tools/python_runtime_probe.py --out DIR [--workers N]
        [--endpoints-json inventory.json] [--public mojolearn.KMeans ...]
        [--group-timeout 600] [--call-timeout 120]
    python3 tools/python_runtime_probe.py --self-test

Outputs under DIR: endpoints.tsv (one row per endpoint), frames.tsv (one row
per endpoint x frame), files.tsv (per-file summary), offenders.tsv (per
(file, function) totals), report.json (counts and the top offenders) and
children/*.jsonl (raw per-public rows, retained). Nothing is printed beyond a
short summary.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / 'python' / 'mojolearn'
SCHEMA = 'mojolearn.python-runtime-probe.v1'
# Verifier fixtures and test helpers are harness code even though they live
# in the package directory.
HARNESS_PREFIXES = ('_verify', '_verification_api_targets', '_causal_lm_fixtures', 'tests/')
SKIP_METHODS = {'__init_subclass__': 'class-creation hook, not a runtime operation',
                '__del__': 'finalizer; runs at collection, not callable as an operation',
                '__exit__': 'context protocol; exercised only with __enter__'}
REFUSALS = ('TypeError', 'ValueError', 'NotImplementedError', 'AttributeError', 'KeyError',
            'IndexError', 'ImportError', 'ModuleNotFoundError', 'NotFittedError', 'LookupError')


# --------------------------------------------------------------------------
# The frame recorder.
# --------------------------------------------------------------------------
class Tracer:
    """sys.setprofile callback. Tracks only frames of an armed call."""

    def __init__(self, product_root, harness_prefixes=HARNESS_PREFIXES):
        self.root = str(product_root) + os.sep
        self.harness = harness_prefixes
        self.state = {}      # id(frame) -> (kind, entry key)
        self.cstack = {}     # id(frame) -> [c function category,...]
        self.frames = Counter()   # (kind, entry, file, func, line) -> count
        self.ccalls = Counter()   # (category, entry, qualname) -> count
        self.entries = []
        self._file_cache = {}

    def classify_file(self, filename):
        hit = self._file_cache.get(filename)
        if hit is None:
            if filename.startswith(self.root):
                rel = filename[len(self.root):]
                hit = ('harness' if rel.startswith(self.harness) else 'product',
                       'python/mojolearn/' + rel)
            else:
                hit = ('other', short_path(filename))
            self._file_cache[filename] = hit
        return hit

    def __call__(self, frame, event, arg):
        try:
            if event == 'call':
                back = frame.f_back
                parent = self.state.get(id(back)) if back is not None else None
                code = frame.f_code
                where, rel = self.classify_file(code.co_filename)
                if parent is not None:
                    kind = 'callback' if self.cstack.get(id(back)) and \
                        self.cstack[id(back)][-1] == 'native' else 'inside'
                    entry = parent[1]
                    self.state[id(frame)] = ('inside', entry)
                    self.frames[(kind, entry, rel, code.co_name, code.co_firstlineno)] += 1
                elif where == 'product':
                    entry = rel + ':' + code.co_name
                    self.state[id(frame)] = ('entry', entry)
                    self.entries.append(entry)
                    self.frames[('entry', entry, rel, code.co_name, code.co_firstlineno)] += 1
            elif event == 'return':
                self.state.pop(id(frame), None)
                self.cstack.pop(id(frame), None)
            elif event == 'c_call':
                st = self.state.get(id(frame))
                if st is not None:
                    category, name = c_category(arg)
                    self.cstack.setdefault(id(frame), []).append(category)
                    self.ccalls[(category, st[1], name)] += 1
            elif event in ('c_return', 'c_exception'):
                stack = self.cstack.get(id(frame))
                if stack:
                    stack.pop()
        except Exception:  # never let the recorder change the probed call
            pass

    def reset(self):
        self.state.clear()
        self.cstack.clear()
        self.frames = Counter()
        self.ccalls = Counter()
        self.entries = []


def short_path(filename):
    for marker in ('site-packages' + os.sep, 'lib' + os.sep + 'python3'):
        i = filename.find(marker)
        if i >= 0:
            return filename[i + len(marker):] if marker.startswith('site') else 'stdlib:' + \
                filename[i + len(marker):].split(os.sep, 1)[-1]
    return filename


def c_category(func):
    name = getattr(func, '__qualname__', None) or getattr(func, '__name__', repr(func))
    owner = getattr(func, '__self__', None)
    module = getattr(func, '__module__', None)
    if module is None and owner is not None:
        module = owner.__name__ if type(owner).__name__ == 'module' else type(owner).__module__
    module = module or ''
    owner_file = getattr(owner, '__file__', '') or '' if type(owner).__name__ == 'module' else ''
    if module.startswith(('mojolearn', '_mojolearn')) or '_mojolearn' in owner_file or \
            ('mojolearn' in module and owner_file.endswith('.so')):
        category = 'native'
    elif module.startswith('numpy'):
        category = 'numpy'
    elif module in ('builtins', ''):
        category = 'builtin'
    else:
        category = 'other'
    return category, (module + '.' if module else '') + name


# --------------------------------------------------------------------------
# Child: one public name.
# --------------------------------------------------------------------------
class CallTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise CallTimeout('per-call timeout')


def run_armed(tracer, operation, call_timeout):
    tracer.reset()
    start = time.monotonic()
    status, error, result_type = 'probed', None, None
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, int(call_timeout)))
    sys.setprofile(tracer)
    try:
        result = operation()
        result_type = type(result).__name__
    except CallTimeout:
        status, error = 'timeout', 'per-call timeout %ss' % call_timeout
    except BaseException as exc:  # noqa: BLE001 -- recorded, never raised
        name = type(exc).__name__
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            sys.setprofile(None)
            signal.alarm(0)
            raise
        status = 'refused' if name in REFUSALS or isinstance(exc, (TypeError, ValueError,
                    NotImplementedError, AttributeError, LookupError, ImportError)) else \
            ('harness_assertion' if isinstance(exc, AssertionError) else 'errored')
        error = (name + ': ' + str(exc))[:400]
    finally:
        sys.setprofile(None)
        signal.alarm(0)
    return dict(status=status, error=error, result_type=result_type,
                seconds=round(time.monotonic() - start, 4),
                frames=[list(k) + [v] for k, v in sorted(tracer.frames.items())],
                ccalls=[list(k) + [v] for k, v in sorted(tracer.ccalls.items())])


def target_names(method):
    """Entry function names that are the endpoint's own public frame."""
    base = method.split(':', 1)[0]
    names = {base}
    if method == 'call':
        names = set()
    if method == '__call__':
        names |= {'forward', '__call__'}
    return names


def child_main(args):
    os.environ.setdefault('MOJOLEARN_VENDOR', 'cpu')
    sys.path.insert(0, str(REPO / 'python'))
    rows = [json.loads(line) for line in Path(args.child_rows).read_text().splitlines() if line]
    public = rows[0]['public']
    out = Path(args.child_out)
    tracer = Tracer(PACKAGE)
    done = set()

    def emit(endpoint, row, record):
        record.update(endpoint=endpoint, public=public, method=row['method'], kind=row['kind'],
                      static_status=row.get('runtime_status'), definition=row.get('definition'))
        with out.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        done.add(endpoint)

    by_endpoint = {r['endpoint']: r for r in rows}
    try:
        import mojolearn  # noqa: F401  (import is not a probed operation)
        from mojolearn import _verify_api as api
    except BaseException as exc:  # noqa: BLE001
        for endpoint, row in by_endpoint.items():
            emit(endpoint, row, dict(status='errored', error='import: ' + type(exc).__name__ + ': ' +
                                     str(exc)[:300], frames=[], ccalls=[], via='import'))
        return 0

    class ProbeRecorder(api.Recorder):
        def __init__(self):
            self.targets = set(by_endpoint)
            self.rows = {}

        def check(self, endpoint, operation, *, kind='numerical'):
            if endpoint not in self.targets or endpoint in done:
                return None
            record = run_armed(tracer, operation, args.call_timeout)
            record['via'] = 'verifier:' + kind
            self.rows[endpoint] = record
            emit(endpoint, by_endpoint[endpoint], record)
            return record

    recorder = ProbeRecorder()
    # 1. The ordinary verifier's fixtures, routed exactly as its registry
    #    routes audit rows (python/mojolearn/_verify_api_runner.py).
    try:
        from mojolearn import (_verify_api_metadata as metadata, _verify_api_estimators as estimators,
            _verify_api_utilities as utilities, _verify_api_optimizers as optimizers,
            _verify_api_neural as neural, _verify_api_training as training,
            _verify_api_programs as programs, _verify_api_models as models, _verify_api_misc as misc)
        families = (('utilities', utilities), ('optimizers', optimizers), ('neural', neural),
                    ('training', training), ('programs', programs), ('models', models),
                    ('estimators', estimators), ('misc', misc))
        groups = defaultdict(list)
        for row in rows:  # driver-loop: over endpoint rows of one public name
            method = row['method']
            if method in metadata.PARAMETER_METHODS:
                groups['metadata'].append(row)
                continue
            for family, module in families:
                if module.applicable(public, [method] if family == 'estimators' else method):
                    groups[family].append(row)
                    break
        for family, members in groups.items():  # driver-loop: over verifier families
            try:
                if family == 'metadata':
                    metadata.parameter_checks(recorder, public)
                elif family == 'utilities':
                    utilities.utility_checks(recorder, [dict(r, op=r['method']) for r in members])
                else:
                    module = dict(families)[family]
                    name = {'estimators': 'estimator', 'optimizers': 'optimizer', 'neural': 'neural',
                            'training': 'training', 'programs': 'program', 'models': 'model',
                            'misc': 'misc'}[family] + '_checks'
                    getattr(module, name)(recorder, public, [r['method'] for r in members])
            except BaseException as exc:  # noqa: BLE001 -- a fixture outside check()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                sys.setprofile(None)
                with out.open('a') as stream:
                    stream.write(json.dumps(dict(fixture_error=family, public=public,
                        error=(type(exc).__name__ + ': ' + str(exc))[:400])) + '\n')
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        sys.setprofile(None)

    # 2. Generic small-input fallback for every endpoint the families left.
    generic_checks(public, [r for r in rows if r['endpoint'] not in done], tracer, emit, args)
    return 0


def tiny_inputs():
    import numpy as np
    X = np.array([[1, 2, 3, 1], [2, 1, 1, 3], [3, 4, 2, 2], [4, 2, 4, 1],
                  [5, 3, 1, 4], [6, 1, 3, 2], [7, 4, 4, 3], [8, 2, 2, 4]], np.float32) / 8
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], np.int64)
    target = np.array([1, 1.25, 1.5, 1.125, 1.75, 1.375, 2, 1.625], np.float32)
    return X, labels, target


def argument(name, X, labels, target, tmp):
    import numpy as np
    n = name.lower()
    if n in ('y', 'y_true', 'labels', 'labels_true', 'labels_pred', 'targets', 'classes', 'groups',
             'y_pred', 'target', 'ids', 'tokens', 'indices'):
        return labels
    if n in ('y_score', 'y_prob', 'probas_pred', 'scores', 'score', 'sample_weight', 'weights', 'w'):
        return np.clip(target - 1, 0, 1).astype(np.float32)
    if n in ('n', 'k', 'n_samples', 'n_components', 'n_clusters', 'size', 'n_features', 'steps',
             'max_norm', 'n_splits', 'dim', 'length', 'seed', 'random_state', 'horizon'):
        return 2
    if n in ('path', 'filename', 'file', 'directory', 'dir', 'checkpoint', 'fname'):
        return os.path.join(tmp, 'probe_artifact')
    if n in ('estimator',):
        import mojolearn
        return mojolearn.Ridge(numeric_mode='identical')
    if n in ('text', 'string', 's'):
        return 'abc'
    if n in ('state', 'params', 'state_dict'):
        return None
    return X


def generic_checks(public, rows, tracer, emit, args):
    import inspect
    rows = [r for r in rows]
    if not rows:
        return
    try:
        from mojolearn._verify_api import resolve
        from mojolearn._verify_api_metadata import factory
        target = resolve(public)
    except BaseException as exc:  # noqa: BLE001
        for row in rows:
            emit(row['endpoint'], row, dict(status='errored', error='resolve: ' + type(exc).__name__ +
                 ': ' + str(exc)[:300], frames=[], ccalls=[], via='generic'))
        return
    X, labels, target_y = tiny_inputs()
    tmp = tempfile.mkdtemp(prefix='probe-')
    obj = {}

    def instance():
        if 'v' not in obj:
            obj['v'] = factory(public)
        return obj['v']

    def fitted():
        m = instance()
        if not obj.get('fitted'):
            obj['fitted'] = True
            fit = getattr(m, 'fit', None)
            if fit is not None:
                try:
                    fit(X, labels if getattr(m, '_estimator_type', '') == 'classifier' else target_y)
                except TypeError:
                    fit(X)
        return m

    def call_with(fn, row):
        params = [p for p in row.get('parameters', []) if p.get('required')
                  and p.get('kind') in ('positional_only', 'positional_or_keyword')
                  and p['name'] not in ('self', 'cls')]
        if params:
            return fn(*[argument(p['name'], X, labels, target_y, tmp) for p in params])
        try:
            return fn()
        except TypeError:
            return fn(X)

    order = sorted(rows, key=lambda r: (r['method'] not in ('__init__', 'call'), r['method'] != 'fit',
                                         r['method']))
    for row in order:  # driver-loop: over endpoints of one public name
        method, kind = row['method'], row['kind']
        if method in SKIP_METHODS or kind.startswith('unresolved'):
            emit(row['endpoint'], row, dict(status='not_probed', error=SKIP_METHODS.get(
                method, 'static inventory could not resolve this name'), frames=[], ccalls=[],
                via='generic'))
            continue
        if not isinstance(target, type):
            op = (lambda: call_with(target, row)) if method == 'call' else None
        elif method == '__init__':
            op = lambda: factory(public)
        elif method.endswith(':get'):
            op = lambda m=method: getattr(fitted(), m[:-4])
        elif method.endswith(':set'):
            def op(m=method):
                inst = fitted()
                value = getattr(inst, m[:-4])
                setattr(inst, m[:-4], value)
                return value
        elif method == '__enter__':
            def op():
                inst = instance()
                with inst as entered:
                    return entered
        elif method in ('__getstate__', '__reduce__', '__reduce_ex__'):
            op = lambda m=method: getattr(fitted(), m)() if m != '__reduce_ex__' else fitted().__reduce_ex__(4)
        elif method == '__repr__':
            op = lambda: repr(instance())
        elif method in ('get_params', 'set_params'):
            op = (lambda: instance().get_params()) if method == 'get_params' else \
                (lambda: instance().set_params(**instance().get_params(deep=False)))
        elif method == '__sklearn_tags__':
            op = lambda: instance().__sklearn_tags__()
        else:
            def op(m=method, r=row):
                inst = instance() if m in ('fit', 'fit_transform', 'fit_predict', 'partial_fit') else fitted()
                attr = getattr(inst, m)
                if not callable(attr):
                    return attr
                return call_with(attr, r)
        if op is None:
            emit(row['endpoint'], row, dict(status='not_probed', error='no generic driver for ' + method,
                                            frames=[], ccalls=[], via='generic'))
            continue
        # Construction/fit needed to reach the endpoint run unarmed first so
        # their frames are not charged to this endpoint.
        if isinstance(target, type) and method not in ('__init__',):
            try:
                if method in ('fit', 'fit_transform', 'fit_predict', 'partial_fit', 'get_params',
                              'set_params', '__repr__', '__enter__', '__sklearn_tags__'):
                    instance()
                else:
                    fitted()
            except BaseException as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                sys.setprofile(None)
        record = run_armed(tracer, op, args.call_timeout)
        record['via'] = 'generic'
        emit(row['endpoint'], row, record)


# --------------------------------------------------------------------------
# Parent: fan out, enforce hard timeouts, aggregate.
# --------------------------------------------------------------------------
def owner_table():
    """{file: owner} from the triage owner table when it is on this machine."""
    owners = {}
    for path in (Path.home() / 'mojolearn-evidence/triage-20261028/prohibited_files_owners_main.tsv',
                 REPO / 'experiments/python_runtime_probe_20261028/owners.tsv'):
        if path.exists():
            for line in path.read_text().splitlines():  # driver-loop: over owner table lines
                cols = line.split('\t')
                if len(cols) >= 2 and cols[0].startswith('python/'):
                    owners.setdefault(cols[0], cols[-1])
            break
    return owners


def run_child(public, rows, out_dir, args, deadline=None):
    if deadline is not None and time.monotonic() > deadline:
        return dict(public=public, exit='deadline', seconds=0, rows=[dict(
            endpoint=r['endpoint'], public=public, method=r['method'], kind=r['kind'],
            static_status=r.get('runtime_status'), definition=r.get('definition'), status='not_probed',
            error='overall probe deadline reached before this public name started', frames=[], ccalls=[],
            via='parent') for r in rows])
    safe = public.replace('/', '_')
    rows_path = out_dir / 'children' / (safe + '.rows.jsonl')
    out_path = out_dir / 'children' / (safe + '.jsonl')
    log_path = out_dir / 'children' / (safe + '.log')
    rows_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if out_path.exists():
        out_path.unlink()
    env = dict(os.environ, MOJOLEARN_VENDOR=os.environ.get('MOJOLEARN_VENDOR', 'cpu'),
               PYTHONPATH=str(REPO / 'python') + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--child-rows', str(rows_path),
               '--child-out', str(out_path), '--call-timeout', str(args.call_timeout)]
    start = time.monotonic()
    with log_path.open('wb') as log:
        proc = subprocess.Popen(command, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=args.group_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = 'timeout'
    got = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():  # driver-loop: over retained child rows
            try:
                got.append(json.loads(line))
            except ValueError:
                pass
    seen = {r.get('endpoint') for r in got}
    for row in rows:  # driver-loop: over endpoints a killed child never reached
        if row['endpoint'] not in seen:
            status = 'timeout' if code == 'timeout' else 'errored'
            got.append(dict(endpoint=row['endpoint'], public=public, method=row['method'], kind=row['kind'],
                            static_status=row.get('runtime_status'), definition=row.get('definition'),
                            status=status, error='child %s before this endpoint (exit %s); see %s' % (
                                'hit the hard group timeout' if code == 'timeout' else 'ended',
                                code, log_path.name), frames=[], ccalls=[], via='parent'))
    return dict(public=public, exit=code, seconds=round(time.monotonic() - start, 2), rows=got)


def aggregate(results, out_dir):
    owners = owner_table()
    endpoint_rows, frame_rows = [], []
    offenders = defaultdict(lambda: dict(calls=0, endpoints=set(), kinds=Counter()))
    files = defaultdict(lambda: dict(calls=0, endpoints=set(), functions=set()))
    status = Counter()
    with_inside = with_inside_target = with_callback = 0
    fixture_errors = []
    for result in results:  # driver-loop: over child results
        for row in result['rows']:
            if 'fixture_error' in row:
                fixture_errors.append(row)
                continue
            status[row['status']] += 1
            targets = target_names(row['method'])
            inside = [f for f in row['frames'] if f[0] in ('inside', 'callback')]
            entry_funcs = {f[3] for f in row['frames'] if f[0] == 'entry'}
            own_entries = {f[1] for f in row['frames'] if f[0] == 'entry' and
                           (not targets or f[3] in targets)}
            inside_target = [f for f in inside if f[1] in own_entries]
            with_inside += bool(inside)
            with_inside_target += bool(inside_target)
            with_callback += any(f[0] == 'callback' for f in row['frames'])
            numpy_calls = sum(c[3] for c in row['ccalls'] if c[0] == 'numpy')
            native_calls = sum(c[3] for c in row['ccalls'] if c[0] == 'native')
            endpoint_rows.append([row['endpoint'], row['kind'], row.get('static_status') or '',
                                  row['status'], row.get('via', ''), len(entry_funcs),
                                  sum(f[5] for f in inside), sum(f[5] for f in inside_target),
                                  sum(f[5] for f in row['frames'] if f[0] == 'callback'),
                                  native_calls, numpy_calls, (row.get('error') or '').replace('\t', ' ')
                                  .replace('\n', ' ')[:200]])
            for f in row['frames']:
                kind, entry, rel, func, line, count = f
                frame_rows.append([row['endpoint'], kind, entry, rel, func, line, count])
                if kind == 'entry':
                    continue
                key = (rel, func)
                offenders[key]['calls'] += count
                offenders[key]['endpoints'].add(row['endpoint'])
                offenders[key]['kinds'][kind] += count
                files[rel]['calls'] += count
                files[rel]['endpoints'].add(row['endpoint'])
                files[rel]['functions'].add(func)

    def owner(rel):
        return owners.get(rel, '')

    def write_tsv(name, header, rows):
        with (out_dir / name).open('w') as stream:
            stream.write('\t'.join(header) + '\n')
            for r in rows:
                stream.write('\t'.join(str(c) for c in r) + '\n')

    write_tsv('endpoints.tsv', ['endpoint', 'kind', 'static_status', 'status', 'via', 'entry_functions',
              'inside_frames', 'inside_frames_own_entry', 'callback_frames', 'native_c_calls',
              'numpy_c_calls', 'error'], sorted(endpoint_rows))
    write_tsv('frames.tsv', ['endpoint', 'kind', 'entry', 'file', 'function', 'first_line', 'count'],
              sorted(frame_rows, key=lambda r: (r[0], r[1], r[3], r[5])))
    ranked = sorted(offenders.items(), key=lambda kv: (-len(kv[1]['endpoints']), -kv[1]['calls'], kv[0]))
    write_tsv('offenders.tsv', ['file', 'function', 'endpoints', 'calls', 'callback_calls', 'owner'],
              [[k[0], k[1], len(v['endpoints']), v['calls'], v['kinds'].get('callback', 0), owner(k[0])]
               for k, v in ranked])
    write_tsv('files.tsv', ['file', 'endpoints', 'calls', 'functions', 'owner'],
              [[k, len(v['endpoints']), v['calls'], len(v['functions']), owner(k)] for k, v in
               sorted(files.items(), key=lambda kv: (-len(kv[1]['endpoints']), kv[0]))])
    report = dict(schema=SCHEMA, scope='CPU column (MOJOLEARN_VENDOR=cpu), numeric_mode=identical, tiny '
                  'verifier fixtures; frames observed, not a proof of absence on other inputs; no identity, '
                  'GPU or timing claim',
                  publics=len(results), endpoints=sum(status.values()), status=dict(status),
                  with_python_inside=with_inside, with_python_inside_own_entry=with_inside_target,
                  with_native_callback=with_callback,
                  child_exits=dict(Counter(str(r['exit']) for r in results)),
                  fixture_errors=len(fixture_errors),
                  top_offenders=[dict(file=k[0], function=k[1], endpoints=len(v['endpoints']),
                                      calls=v['calls'], owner=owner(k[0])) for k, v in ranked[:25]])
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    return report


def load_endpoints(args, out_dir):
    path = args.endpoints_json
    if path is None:
        path = out_dir / 'inventory-endpoints.json'
        with (out_dir / 'inventory.log').open('wb') as log:
            code = subprocess.call([sys.executable, str(REPO / 'tools/public_runtime_inventory.py'),
                                    '--endpoints-json', str(path)], cwd=str(REPO), stdout=log,
                                   stderr=subprocess.STDOUT)
        if not path.exists():
            raise SystemExit('inventory scan failed (exit %s); see inventory.log' % code)
    return json.loads(Path(path).read_text())['endpoints']


def self_test():
    """The recorder on a synthetic product package; no mojolearn import."""
    root = Path(tempfile.mkdtemp(prefix='probe-selftest-'))
    (root / 'pkg').mkdir()
    (root / 'pkg' / 'api.py').write_text(
        'import json\n'
        'def helper(v):\n    return [x * 2 for x in v]\n'
        'def clean(v):\n    return sorted(v)\n'
        'def dirty(v):\n    return helper(v)\n'
        'def via_c(v):\n    return list(map(helper, [v]))\n')
    sys.path.insert(0, str(root))
    import importlib
    api = importlib.import_module('pkg.api')
    tracer = Tracer(root / 'pkg')

    def harness_clean():
        return api.clean([3, 1, 2])

    def harness_dirty():
        return api.dirty([1, 2])

    def harness_via_c():
        return api.via_c([1])
    clean = run_armed(tracer, harness_clean, 5)
    dirty = run_armed(tracer, harness_dirty, 5)
    via = run_armed(tracer, harness_via_c, 5)
    kinds = lambda r: sorted({(f[0], f[3]) for f in r['frames']})
    assert kinds(clean) == [('entry', 'clean')], kinds(clean)
    assert ('inside', 'helper') in kinds(dirty), kinds(dirty)
    assert ('inside', '<listcomp>') in kinds(dirty) or sys.version_info >= (3, 12), kinds(dirty)
    assert ('inside', 'helper') in kinds(via), kinds(via)  # map is builtin, not native
    assert any(c[0] == 'builtin' for c in clean['ccalls']), clean['ccalls']
    refused = run_armed(tracer, lambda: api.clean(None), 5)
    assert refused['status'] == 'refused', refused
    print('self-test ok')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--endpoints-json', type=Path)
    parser.add_argument('--public', action='append', default=[])
    parser.add_argument('--workers', type=int, default=0, help='default: allocated cores')
    parser.add_argument('--group-timeout', type=float, default=600.0, help='hard, per public name')
    parser.add_argument('--call-timeout', type=float, default=120.0, help='per probed call')
    parser.add_argument('--deadline-seconds', type=float, default=0.0,
                        help='stop starting children after this many seconds (rows become not_probed)')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='rebuild the reports from retained children/*.jsonl (e.g. to attach owners)')
    parser.add_argument('--child-rows', help=argparse.SUPPRESS)
    parser.add_argument('--child-out', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.child_rows:
        return child_main(args)
    if args.out is None:
        parser.error('--out is required')
    out_dir = args.out.resolve()
    if args.aggregate_only:
        results = []
        for path in sorted((out_dir / 'children').glob('*.jsonl')):  # driver-loop: over retained files
            if path.name.endswith('.rows.jsonl'):
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            results.append(dict(public=path.name[:-6], exit='retained', seconds=0, rows=rows))
        previous = out_dir / 'report.json'
        kept = json.loads(previous.read_text()) if previous.exists() else {}
        report = aggregate(results, out_dir)
        for key in ('workers', 'seconds', 'child_exits'):
            if key in kept:
                report[key] = kept[key]
        (out_dir / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
        return 0
    (out_dir / 'children').mkdir(parents=True, exist_ok=True)
    endpoints = load_endpoints(args, out_dir)
    grouped = defaultdict(list)
    for row in endpoints:  # driver-loop: over inventory endpoint rows
        if not args.public or row['public'] in args.public:
            grouped[row['public']].append(row)
    workers = args.workers or len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else \
        (args.workers or os.cpu_count() or 1)
    start = time.monotonic()
    deadline = start + args.deadline_seconds if args.deadline_seconds > 0 else None
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_child, public, rows, out_dir, args, deadline)
                   for public, rows in sorted(grouped.items())]
        for future in futures:  # driver-loop: over child processes
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(dict(public='?', exit='parent-error', seconds=0, rows=[],
                                    error=traceback.format_exc()[-400:]))
                print('parent error:', type(exc).__name__, file=sys.stderr)
    (out_dir / 'children.json').write_text(json.dumps(
        [dict(public=r['public'], exit=r['exit'], seconds=r['seconds'], rows=len(r['rows'])) for r in results],
        indent=1) + '\n')
    report = aggregate(results, out_dir)
    report.update(workers=workers, seconds=round(time.monotonic() - start, 1))
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('publics', 'endpoints', 'status', 'with_python_inside',
                      'with_python_inside_own_entry', 'with_native_callback', 'child_exits', 'workers',
                      'seconds')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
