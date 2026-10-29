#!/usr/bin/env python3
"""Compile-only receipt and retained source bytes; never executes the artifact."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAIN = pathlib.Path('/Users/andrewhendel/CascadeProjects/mojolearn')
variant = 'native'
OUT = pathlib.Path.home() / 'mojolearn-evidence/resident-layer-workspaces-20261029' / variant
OUT.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
files = [name for name in files if name.endswith('.mojo') or name in ('pixi.toml', 'pixi.lock')]
files += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
files = sorted({name for name in files if name.endswith('.mojo') or name in ('pixi.toml', 'pixi.lock')})
manifest = {name: sha(ROOT / name) for name in files}
(OUT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with tarfile.open(OUT / 'source-bytes.tgz', 'w:gz') as archive:
    for name in files:
        archive.add(ROOT / name, arcname=name)
workers = int(subprocess.check_output(['sysctl', '-n', 'hw.logicalcpu']))
compiler = MAIN / '.pixi/envs/default/bin/mojo'
artifact = OUT / '_resident_layer_workspace.so'
command = ['bash', 'tools/compile_slot.sh', 'pixi', 'run', '--manifest-path', str(MAIN / 'pixi.toml'),
           '--environment', 'default', '--locked', '--no-install', '--executable', str(compiler),
           'build', '--optimization-level', '0', '-j', str(workers), '--emit', 'shared-lib',
           '--target-cpu', 'apple-m1', '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1', 
           '-I', str(ROOT), '-I', str(ROOT / 'bindings'), str(ROOT / 'experiments/resident_layer_workspaces_20261029/compile.mojo'),
           '-o', str(artifact)]
start = time.monotonic()
with (OUT / 'compile.log').open('w') as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(exit_code=result.returncode, seconds=time.monotonic()-start, command=command,
               base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
               cpu_allocation=workers, workers=workers, allocation_source='Mac sysctl hw.logicalcpu; compile_slot wrapper; concurrent peers may exist',
               scope='Actual native resident LM dispatch including layer prefill, forward/run/read/reservation lifecycle; GPU identical O0 on Apple CPU host (no COLUMN_CPU)',
               runtime_executed=False, compiler_sha256=sha(compiler),
               log=str(OUT/'compile.log'), artifact=str(artifact),
               source_archive=str(OUT/'source-bytes.tgz'), source_archive_sha256=sha(OUT/'source-bytes.tgz'),
               source_manifest=str(OUT/'source-manifest.json'),
               pending=['Runtime, identity and performance deliberately not run', 'NVIDIA/AMD target compilation not covered',
                        'Unchanged public boundary prior compile reused; no installed/runtime dispatch claim'])
if result.returncode == 0:
    receipt['artifact_sha256'] = sha(artifact)
for path in (OUT/'compile.json', ROOT/('experiments/resident_layer_workspaces_20261029/compile_' + variant.replace('-', '_') + '.json')):
    path.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({key: receipt[key] for key in ('exit_code', 'seconds', 'scope', 'log')}))
raise SystemExit(result.returncode)
