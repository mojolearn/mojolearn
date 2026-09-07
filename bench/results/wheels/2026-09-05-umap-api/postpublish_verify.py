import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request

repo = Path.cwd()
out = repo / 'bench/results/wheels/2026-09-05-umap-api/postpublish'
out.mkdir(exist_ok=True)
filename = 'mojolearn-0.5.0-py3-none-macosx_11_0_arm64.whl'
built = Path.home() / '.mojolearn-runner/_work/mojolearn/mojolearn/python/dist' / filename
expected = (out.parent / 'publication-expected.sha256').read_text().split()[0]
with urllib.request.urlopen('https://pypi.org/pypi/mojolearn/0.5.0/json', timeout=30) as response:
    metadata = json.load(response)
files = [f for f in metadata['urls'] if f['filename'] == filename]
assert len(files) == 1 and files[0]['digests']['sha256'] == expected
wheel = out / filename
with urllib.request.urlopen(files[0]['url'], timeout=120) as response:
    wheel.write_bytes(response.read())
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected
record = {'version': '0.5.0', 'filename': filename, 'sha256': expected,
          'url': files[0]['url'], 'pypi_matches_publication_build': True,
          'source_commit': '529ec5ec73c81c14826135fe615fac1a7631bf54', 'results': []}
env = os.environ.copy()
for key in ['PYTHONPATH', 'PYTHONHOME', 'MOJOLEARN_NUMERIC_MODE']:
    env.pop(key, None)
with tempfile.TemporaryDirectory(prefix='mojolearn-pypi-verify-') as tmp:
    venv = Path(tmp) / 'venv'
    subprocess.run(['/opt/homebrew/bin/python3.12', '-m', 'venv', str(venv)], check=True, env=env)
    python = str(venv / 'bin/python')
    with (out / 'setup.log').open('w') as log:
        subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', str(wheel)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([python, '-c', 'import mojolearn; print(mojolearn.__version__, mojolearn.__file__); assert mojolearn.__version__ == "0.5.0"; assert "site-packages" in mojolearn.__file__'], cwd=tmp, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    for mode in ['fast', 'deterministic', 'identical']:
        env['MOJOLEARN_NUMERIC_MODE'] = mode
        for name, script, cwd in [
                ('smoke', repo / 'packaging/macos/smoke.py', tmp),
                ('mamba', repo / 'python/mojolearn/tests/test_mamba_surface.py', repo),
                ('transformer', repo / 'python/mojolearn/tests/test_transformer_surface.py', repo)]:
            print('RUN PyPI', name, mode, flush=True)
            logname = f'{name}-{mode}.log'
            with (out / logname).open('w') as log:
                result = subprocess.run([python, str(script)], cwd=cwd, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=600)
            record['results'].append({'surface': name, 'mode': mode,
                                      'exit_code': result.returncode, 'log': logname})
            (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
            print('EXIT', result.returncode, name, mode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
print('PYPI ARTIFACT AND INSTALLED API PASS', expected, flush=True)
