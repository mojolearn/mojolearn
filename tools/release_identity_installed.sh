#!/usr/bin/env bash
# On-box artifact identity gate. No rental, native build, benchmark or teardown.
# Run INSIDE the existing GPU scheduler:
#   release_identity_installed.sh CORE SHA VENDOR OUT COMMIT ARCH SELECTION [PLUGIN...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec "${MOJOLEARN_QUALIFY_PYTHON:-python3}" - "$ROOT" "$@" <<'PY'
# IDENTITY_PY_BEGIN
import hashlib, json, os, pathlib, re, signal, subprocess, sys, time, zipfile
P = pathlib.Path
PARTS = ('train', 'infer', 'model', 'batch', 'stepfull')

def sha(path):
    return hashlib.sha256(P(path).read_bytes()).hexdigest()

def admit_identity(doc, lanes):
    assert doc['format'] == 'mojolearn.verify-all-report.v1'
    assert (doc['exit'], doc['verdict']) == (0, 'VERIFIED')
    assert doc['fixtures'] == ['base'] and doc['repeats'] == 1
    assert len(doc['lanes']) == len(lanes) and set(doc['lanes']) == set(lanes)
    assert doc['models_checked'] == 0
    execution = doc['execution']
    assert execution['mode'] == 'fresh-process-per-cell'
    assert execution['timeout_s'] == 120 and execution['interrupted'] is None
    assert execution['completed_cells'] == execution['total_cells'] == len(lanes)
    expected = {(lane, 'base', part) for lane in lanes for part in PARTS}
    rows = doc['cells']
    keys = [(row['lane'], row['fixture'], row['part']) for row in rows]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    for row in rows:
        assert row['state'] in ('IDENTICAL', 'N/A'), row
        assert not row.get('error'), row
        if row['state'] == 'IDENTICAL':
            assert re.fullmatch('[0-9a-f]{16}', row['value']), row
        else:
            assert isinstance(row.get('value'), str) and row['value'].startswith('n/a:'), row
            assert row['value'] not in ('n/a:UNDECLARED', 'n/a:skipped'), row
    assert doc.get('bindings') and not doc.get('bindings_error')

def main(argv):
    if not __debug__: raise RuntimeError('Python -O is refused for qualification')
    if len(argv) < 8:
        raise SystemExit('Expected ROOT CORE SHA VENDOR OUT COMMIT ARCH SELECTION [PLUGIN...]')
    root, wheel, expected_sha, vendor, out, commit, arch, selection, *plugins = argv
    root, wheel, out, selection = map(lambda p: P(p).resolve(), (root, wheel, out, selection))
    plugins = [P(p).resolve() for p in plugins]
    assert vendor in ('cuda', 'hip') and arch in ('sm_89', 'sm_90', 'sm_90a', 'gfx942')
    assert vendor == ('hip' if arch == 'gfx942' else 'cuda')
    assert re.fullmatch('[0-9a-f]{40}', commit) and sha(wheel) == expected_sha
    with zipfile.ZipFile(wheel) as z:
        assert z.read('mojolearn/identity_columns/COMMIT').decode().strip() == commit
    lanes = json.loads(selection.read_text())['lanes']
    assert lanes and len(lanes) == len(set(lanes))
    assert all(re.fullmatch('[a-z0-9][a-z0-9_.-]*', n) for n in lanes)
    assert not out.exists(), 'Use a fresh output directory; failed evidence is retained'
    out.mkdir(parents=True)
    (out/'exit_code').write_text('1\n')
    limit = int(os.environ.get('MOJOLEARN_IDENTITY_SECONDS', '3600'))
    assert 120 <= limit <= 7200
    env = {k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_')
           and k not in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE','LD_LIBRARY_PATH')}
    env.update(MOJOLEARN_NUMERIC_MODE='identical', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MOJOLEARN_CPU_THREADS='1')
    receipt = dict(schema='mojolearn.installed-light-identity.v1', status='INCOMPLETE',
        source_commit=commit, vendor=vendor, architecture=arch,
        artifacts=[dict(name=p.name, sha256=sha(p)) for p in [wheel]+plugins],
        selection_sha256=sha(selection), lanes=lanes, fixtures=['base'], repeats=1,
        parts=list(PARTS), excluded_properties=['batchgrad','batchscale','ragged','rlpair'],
        performance_benchmarks=False, whole_surface_suite=False,
        tooling_sha256={name:sha(root/'tools'/name) for name in
                        ('release_identity_installed.sh','qualify_verifier_wheel.py')}, jobs=[])
    def save():
        (out/'identity-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    def run(name, cmd, seconds=180):
        started=time.monotonic(); timed=False
        with (out/(name+'.log')).open('w') as f:
            p=subprocess.Popen(list(map(str,cmd)), cwd=out, env=env, stdin=subprocess.DEVNULL,
                               stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try: code=p.wait(timeout=seconds)
            except subprocess.TimeoutExpired: timed=True; code=124
            finally:
                try: os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                p.wait()
        receipt['jobs'].append(dict(name=name, command=list(map(str,cmd)), exit=code,
                                   timeout=timed, seconds=round(time.monotonic()-started,3)))
        save()
        if code: raise RuntimeError(f'{name} failed ({code}); original log retained')
    save()
    try:
        # Retain the existing independently judged smoke receipt unchanged.
        cmd=[sys.executable,root/'tools/qualify_verifier_wheel.py',wheel,'--scope','expanded',
             '--python',sys.executable,'--expected-source-commit',commit,'--output',out/'smoke']
        for p in plugins: cmd += ['--plugin',p]
        run('expanded-smoke',cmd,1800)
        smoke=json.loads((out/'smoke/results.json').read_text())
        assert smoke['status']=='PASSED' and smoke['scope']=='expanded'
        assert smoke['wheel_sha256']==expected_sha and smoke['source_commit']==commit
        assert smoke['installed']['vendor']==vendor
        assert {x['wheel_sha256'] for x in smoke.get('plugins',[])}=={sha(p) for p in plugins}
        run('create-venv',[sys.executable,'-m','venv',out/'venv'],60)
        py=out/'venv/bin/python'
        run('install',[py,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:',
                       wheel,*plugins,'numpy>=1.24'],180)
        run('pip-check',[py,'-m','pip','check'])
        guard="""import pathlib,sys,json,mojolearn as m
from mojolearn import _backend
p=pathlib.Path(m.__file__).resolve();assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()) and 'site-packages' in p.parts
assert m.vendor()==sys.argv[1] and m.numeric_mode()=='identical'
assert m.__version__==sys.argv[3]
assert (p.parent/'identity_columns/COMMIT').read_text().strip()==sys.argv[4]
assert _backend.gpu_arch()==sys.argv[2]
print(json.dumps(dict(package=str(p),vendor=m.vendor(),architecture=_backend.gpu_arch(),version=m.__version__)))
"""
        run('installed-guard',[py,'-c',guard,vendor,arch,wheel.name.split('-')[1],commit])
        # Explicit selected lanes intentionally request the five core parts.
        # --all would also request extended gradient/ragged/replay properties.
        run('identity',[py,'-m','mojolearn','verify','--lanes',','.join(lanes),
            '--fixtures','base','--repeats','1','--no-models','--cpu-threads','1',
            '--cell-timeout','120','--json-out',out/'identity.json','--json'],limit)
        admit_identity(json.loads((out/'identity.json').read_text()),lanes)
        assert all(sha(p)==r['sha256'] for p,r in zip([wheel]+plugins,receipt['artifacts']))
        receipt.update(status='PASSED', identity_sha256=sha(out/'identity.json'),
                       smoke_receipt_sha256=sha(out/'smoke/results.json'))
        save();(out/'exit_code').write_text('0\n')
    except Exception as exc:
        receipt.update(status='FAILED', reason=str(exc));save();raise
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':
    main(sys.argv[1:])
# IDENTITY_PY_END
PY
