from pathlib import Path
import hashlib, importlib.util, json, os, signal, subprocess, sys, time

base = Path(__file__).resolve().parent
ordinary = '--ordinary' in sys.argv
classical = '--classical' in sys.argv
artifact = base/'ordinary' if ordinary else base/'classical-extended' if classical else base
artifact.mkdir(exist_ok=True)
root = Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
spec = importlib.util.spec_from_file_location('capture', root/'tools/capture_ordinary_holds.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
python = str(base/'venv/bin/python')
wheel = next((base/'dist').glob('*.whl'))
env = {k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'DYLD_')) and k not in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE')}
env.update(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_CPU_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONNOUSERSITE='1')
package = Path(subprocess.check_output([python,'-c','import mojolearn; print(mojolearn.__file__)'],cwd=base,env=env,text=True).strip()).parent
receipt = capture.verify_wheel(wheel, package)
receipt.update(scope='Watched installed development CPU replay; mixed retained native binaries, no release qualification', results={}, problems=[])
if (artifact/'replay-receipt.json').exists():
    previous=json.loads((artifact/'replay-receipt.json').read_text())
    if previous['wheel_sha256'] != receipt['wheel_sha256']:
        raise RuntimeError('resume wheel changed')
    receipt=previous
if 'site-packages' not in str(package):
    raise RuntimeError('not an isolated installed package')
def save():
    capture.atomic_json(artifact/'replay-receipt.json',receipt)
save()
cases = [('models',['--models-only','--repeats','2']), ('self-test',['--self-test'])]
cases += [(lane,['--all','--include-pending','--lanes',lane,'--repeats','2',*(['--batch-checks'] if not ordinary else []),'--no-models']) for lane in (capture.LANES[5:] if classical else capture.LANES)]
for name,args in cases:
    if name in receipt['results']:
        continue
    started = time.monotonic()
    command = [python,'-m','mojolearn','verify',*args,'--json']
    with (artifact/(name+'.json')).open('w') as out, (artifact/(name+'.log')).open('w') as err:
        p = subprocess.Popen(command,cwd=base,env=env,stdout=out,stderr=err,start_new_session=True)
        def terminate(signum, frame):
            os.killpg(p.pid, signal.SIGTERM)
            raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        try:
            code = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            code = 124
    result = dict(command=command, exit=code, seconds=round(time.monotonic()-started,3))
    try:
        doc=json.loads((artifact/(name+'.json')).read_text())
        result.update(counts=doc.get('counts'), verdict=doc.get('verdict'), models_checked=doc.get('models_checked'), fixtures=doc.get('fixtures'), properties=doc.get('properties'), scope_gaps=doc.get('scope_gaps'))
        if name in ('models','self-test'):
            if code != 0: receipt['problems'].append(name)
        elif code not in (0,5) or any(doc['counts'].get(k,0) for k in ('REFUSED','DIVERGENT','OWED')):
            receipt['problems'].append(name)
    except (ValueError,KeyError):
        receipt['problems'].append(name)
    receipt['results'][name]=result
    save()
    print(name,json.dumps(result),flush=True)
receipt['status']='PASSED_DEVELOPMENT_CPU_REPLAY' if not receipt['problems'] else 'GAPS_REMAIN'
save()
sys.exit(bool(receipt['problems']))
