#!/usr/bin/env python3
"""compile-fix driver (runs on the 4090 box).

cf_driver.py setup <sha>               -> /root/lq/compile-fix-<sha12>/source worktree
cf_driver.py run <sha> <jobs.tsv> [P]  -> compile each job (4 slots via compile_slot.sh), -j 1
jobs.tsv columns: name  root  tier  arch  arm      (arch = cpu for host column; arm = on|off)
Writes logs/<key>.log and appends results.tsv: name target tier arm rc seconds first_error
"""
import os, subprocess, sys, time, pathlib, threading, re
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path('/root/mojolearn')
ENVROOT = ROOT / '.pixi/envs/default'
MOJO = ENVROOT / 'bin/mojo'
SEM = '/root/mojolearn-evidence/compile_slot.sh'


def outdir(sha):
    return pathlib.Path('/root/lq') / ('br2-compile-' + sha[:12])


def setup(sha):
    od = outdir(sha); src = od / 'source'
    od.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        subprocess.run(['git', '-C', str(ROOT), 'fetch', '-q', 'origin', 'lane/box-run-2'], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '-f', '--detach', str(src), sha], check=True)
    # the tokenizer's Unicode table is generated, not tracked (build_host_family.sh does the same)
    env = dict(os.environ, PATH=str(ENVROOT / 'bin') + ':' + os.environ['PATH'])
    subprocess.run(['sh', 'tokenizer/tools/gen_unicode_table.sh'], cwd=src, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    print('ready', src)


LOCK = threading.Lock()


def run_job(sha, job):
    name, root, tier, arch, arm = job
    od = outdir(sha); src = od / 'source'
    key = f'{name}-{tier}-{arch}-{arm}'
    (od / 'logs').mkdir(exist_ok=True); (od / 'so').mkdir(exist_ok=True)
    log = od / 'logs' / (key + '.log')
    defines = []
    if arch == 'cpu':
        defines += ['MOJOLEARN_NUMERIC_IDENTICAL=1', 'MOJOLEARN_COLUMN_CPU']
        tflags = []
    else:
        defines += ['MOJOLEARN_COLUMN_' + ('NVIDIA' if arch.startswith('sm_') else 'AMD')]
        if tier == 'identical':
            defines += ['MOJOLEARN_NUMERIC_IDENTICAL=1']
        elif tier == 'deterministic':
            defines += ['MOJOLEARN_NUMERIC_DETERMINISTIC=1']
        tflags = ['--target-accelerator', arch]
    if arm == 'off':
        defines += ['MOJOLEARN_IDN_ALL_OFF=1']
    cmd = ['bash', SEM, str(MOJO), 'build', '-j', '1', '--emit', 'shared-lib'] + tflags
    for d in defines:
        cmd += ['-D', d]
    cmd += ['-I', str(src), '-I', str(src / 'bindings'), str(src / root), '-o', str(od / 'so' / (key + '.so'))]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_', 'MOJO_COMPILE_'))}
    env.update(CONDA_PREFIX=str(ENVROOT), MODULAR_HOME=str(ENVROOT / 'share/max'), MOJOLEARN_COMPILE_JOBS='1',
               PATH=str(ENVROOT / 'bin') + ':' + env['PATH'])
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('COMMAND ' + ' '.join(cmd) + '\n'); f.flush()
        try:
            rc = subprocess.run(cmd, cwd=src, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=5400).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    secs = round(time.time() - t0, 1)
    first = ''
    for ln in open(log, errors='replace'):
        if re.search(r'error:', ln):
            first = ln.strip().replace(str(src) + '/', '')[:300]
            break
    (od / 'so' / (key + '.so')).unlink(missing_ok=True)
    with LOCK:
        with open(od / 'results.tsv', 'a') as f:
            f.write('\t'.join([name, arch, tier, arm, str(rc), str(secs), first.replace('\t', ' ')]) + '\n')
    return rc


def run(sha, jobs_path, par=4):
    jobs = [l.rstrip('\n').split('\t') for l in open(jobs_path) if l.strip() and not l.startswith('#')]
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda j: run_job(sha, j), jobs))
    (outdir(sha) / ('DONE-' + pathlib.Path(jobs_path).name)).write_text('done\n')


if __name__ == '__main__':
    if sys.argv[1] == 'setup':
        setup(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
