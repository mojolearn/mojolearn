"""Apple counterpart of tools/knn_layout_dispatch_price.sh; no coreutils required."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
os.chdir(ROOT)
ARMS = ('baseline', 'selector', 'transpose', 'both')
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MOJOLEARN_NUMERIC_MODE='identical')
DEADLINE = time.monotonic() + 1200
STATUS = OUT / 'status.tsv'
STATUS.touch(exist_ok=False)


def run(name, command, env=ENV):
    remaining = DEADLINE - time.monotonic()
    code = 124
    if remaining > 0:
        with (OUT / (name + '.log')).open('w') as log:
            p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                code = p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
    with STATUS.open('a') as status:
        status.write(f'{name}\t{code}\n')
    print(name, code, flush=True)
    if code:
        raise SystemExit(code)


sources = subprocess.check_output(['git', 'ls-files', 'bench/*.mojo', 'neighbors', 'core',
                                  'checks', 'gemm', 'pixi.toml', 'pixi.lock'], text=True).splitlines()
metadata = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            'scope': 'native-public-upload-search-download-synchronize',
            'orchestrator': 'run.py; same builds, warmups and rotating invocation order as shell campaign',
            'work_timeout_seconds': 1200, 'compile_jobs': 2}
(OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
(OUT / 'commit.txt').write_text(metadata['commit'] + '\n')
with tempfile.TemporaryDirectory(prefix='mojolearn-layout-apple-') as bins:
    binaries = {}
    for arm in ARMS:
        flags = ['-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1']
        if arm in ('selector', 'both'):
            flags += ['-D', 'MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1']
        if arm in ('transpose', 'both'):
            flags += ['-D', 'MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1']
        for kind in ('check', 'price'):
            binary = str(Path(bins) / (kind + '-' + arm))
            run(f'build-{kind}-{arm}', ['pixi', 'run', 'mojo', 'build', '-j', '2', '-I', '.',
                *flags, f'bench/knn_layout_dispatch_{kind}.mojo', '-o', binary])
            binaries[kind + '-' + arm] = binary
        run('check-' + arm, [binaries['check-' + arm]])
    (OUT / 'binary-sha256.json').write_text(json.dumps({
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in binaries.items()
    }, indent=2) + '\n')
    for q in (32, 128, 1000):
        for r in range(9):
            for offset in range(4):
                arm = ARMS[(r + offset) % 4]
                run(f'q{q}-r{r}-{arm}', [binaries['price-' + arm]],
                    dict(ENV, MOJOLEARN_SMALLK_PRICE_QUERIES=str(q)))
(OUT / 'completion.txt').write_text('COMPLETE\n')
