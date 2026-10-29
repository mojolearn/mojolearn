"""Focused compile-only integration. Never imports or executes product outputs."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import time

root = Path.cwd()
out = Path(os.environ['LEG_OUT']) / 'focused-compiles'
out.mkdir(parents=True, exist_ok=True)
affinity = sorted(os.sched_getaffinity(0))
limits = []
mount = Path('/sys/fs/cgroup')
paths = {mount / 'cpu.max'}
for line in Path('/proc/self/cgroup').read_text().splitlines():
    if line.startswith('0::'):
        relative = line[3:].lstrip('/')
        current = mount / relative
        if current.is_dir():
            while current == mount or mount in current.parents:
                paths.add(current / 'cpu.max')
                if current == mount:
                    break
                current = current.parent
for path in sorted(paths):
    if not path.exists():
        continue
    words = path.read_text().split()
    limits.append({'path': str(path), 'value': ' '.join(words)})
    if words[0] != 'max':
        limits[-1]['cores'] = max(1, int(words[0]) // int(words[1]))
v1 = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
if not limits and v1.exists():
    quota = int(v1.read_text())
    period = int(v1.with_name('cpu.cfs_period_us').read_text())
    limits.append({'path': str(v1), 'quota': quota, 'period': period})
    if quota > 0:
        limits[-1]['cores'] = max(1, quota // period)
workers = min([len(affinity)] + [x['cores'] for x in limits if 'cores' in x])
env = os.environ.copy()
for key in ('MOJOLEARN_BUILD_JOBS', 'MOJOLEARN_COMPILE_JOBS', 'MOJOLEARN_COMPILER_WORKERS'):
    env[key] = str(workers)
resources = dict(affinity=affinity, cgroup_limits=limits, compiler_workers=workers,
                 concurrent_compiles=1, policy='full actual allocation; saturation unmeasured')
(out / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
compiler = root / '.pixi/envs/default/bin/mojo'
version = subprocess.check_output([str(compiler), '--version'], text=True).strip()
with (out / 'unicode.log').open('w') as log:
    generated = subprocess.run(['bash', 'tokenizer/tools/gen_unicode_table.sh'],
                               stdout=log, stderr=subprocess.STDOUT, env=env)
if generated.returncode:
    raise SystemExit(generated.returncode)

# Conservative source superset; actual compiled scope is the five roots below.
files = sorted(p for p in root.rglob('*.mojo') if '.pixi' not in p.parts and '.git' not in p.parts)
files += [root / 'pixi.toml', root / 'pixi.lock', Path(__file__).resolve()]
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tarfile.open(out / 'source-bytes.tgz', 'w:gz') as archive:
    for p in files:
        archive.add(p, arcname=str(p.relative_to(root)))
roots = {
    'model-selection': 'experiments/structural_reuse_fourth_20261028/compile_model_selection.mojo',
    'attributes': 'experiments/structural_reuse_fourth_20261028/compile_attributes.mojo',
    'tokenizer': 'experiments/structural_reuse_fourth_20261028/compile_tokenizer.mojo',
    'samba': 'experiments/samba_native_connections_20261028/compile.mojo',
    'distributed': 'experiments/distributed_replay_reuse_20261028/compile.mojo',
}
results = []
for name, fixture in roots.items():
    binary = out / (name + '.so')
    command = ['pixi', 'run', '--manifest-path', str(root / 'pixi.toml'),
               '--environment', 'default', '--locked', '--no-install', '--executable',
               str(compiler), 'build', '-j', str(workers), '--emit', 'shared-lib',
               '--optimization-level', '0', '--target-cpu', 'x86-64-v3',
               '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1', '-D', 'MOJOLEARN_COLUMN_CPU',
               '-I', '.', '-I', 'bindings', fixture, '-o', str(binary)]
    start = time.monotonic()
    with (out / (name + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    receipt = dict(name=name, root=fixture, command=command, exit_code=result.returncode,
                   seconds=time.monotonic() - start,
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None)
    results.append(receipt)
    (out / (name + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(name, 'exit', result.returncode, flush=True)
summary = dict(source_commit=os.environ.get('MOJOLEARN_COMMIT'), compiler=version,
               platform=platform.platform(), resources=resources, results=results,
               runtime_executed=False, identity_verified=False, performance_measured=False)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
raise SystemExit(1 if any(r['exit_code'] for r in results) else 0)
