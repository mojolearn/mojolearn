import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

repo = Path.cwd()
out = repo / 'bench/results/wheels/2026-09-05-umap-api'
source = Path.home() / '.mojolearn-runner/_work/mojolearn/mojolearn/python/dist/mojolearn-0.5.0-py3-none-macosx_11_0_arm64.whl'
wheel = out / source.name
shutil.copy2(source, wheel)
sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
(out / 'qualified-wheel.sha256').write_text(sha + '  ' + wheel.name + '\n')
env = os.environ.copy()
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
results = []
with tempfile.TemporaryDirectory(prefix='mojolearn-api-qualification-') as tmp:
    venv = Path(tmp) / 'venv'
    subprocess.run(['/opt/homebrew/bin/python3.12', '-m', 'venv', str(venv)], check=True, env=env)
    python = str(venv / 'bin/python')
    with (out / 'installed-api-setup.log').open('w') as log:
        subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', str(wheel)], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([python, '-c', 'import mojolearn; print(mojolearn.__version__, mojolearn.__file__); assert "site-packages" in mojolearn.__file__'], cwd=tmp, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    for mode in ['fast', 'deterministic', 'identical']:
        env['MOJOLEARN_NUMERIC_MODE'] = mode
        for name in ['mamba', 'transformer']:
            filename = f'installed-{name}-{mode}.log'
            print('RUN', name, mode, flush=True)
            with (out / filename).open('w') as log:
                result = subprocess.run([python, str(repo / f'python/mojolearn/tests/test_{name}_surface.py')], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            results.append({'surface': name, 'mode': mode, 'exit_code': result.returncode, 'log': filename})
            (out / 'installed-api-results.json').write_text(json.dumps({'wheel_sha256': sha, 'results': results}, indent=2) + '\n')
            print('EXIT', result.returncode, name, mode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
