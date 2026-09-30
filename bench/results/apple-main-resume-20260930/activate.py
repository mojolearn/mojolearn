import fcntl, hashlib, json, os, pathlib, shlex, subprocess, time, zipfile

B = pathlib.Path(__file__).resolve().parent
lock = (B / 'activate-apple-main.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(0)
if (B / 'apple-main-activated.json').exists():
    raise SystemExit(0)
commit = '10523ede164135c2293d16f857b49c157eab71aa'
root = '/Users/ec2-user/board-resume-main-10523ede1'
job = '1790789447486-speed-apple-main-wheel-10523ede16'
hosts = {'m3ultra-b': '54.157.1.251', 'm2pro': '54.237.205.45'}
key = str(pathlib.Path.home() / '.ssh/mambik-l8.pem')
ssh_opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-i', key]

def run(args, **kw):
    p = subprocess.run(args, capture_output=True, text=True, timeout=1800, **kw)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:] or p.stdout[-2000:])
    return p.stdout

def remote(host, command, stdin=None):
    return run(['ssh', *ssh_opts, 'ec2-user@' + host, command], input=stdin)

status = remote(hosts['m3ultra-b'], 'python3 -', '''import pathlib,json
p=pathlib.Path.home()/'mojolearn-evidence/apple-steward/done'/''' + repr(job) + '''/'verdict.json'
print(p.read_text() if p.exists() else '{}')
''')
verdict = json.loads(status)
if not verdict:
    print('Wheel build still running', flush=True)
    raise SystemExit(0)
if verdict.get('result') != 'PASS':
    (B / 'apple-main-build-failure.json').write_text(json.dumps(verdict, indent=2))
    raise SystemExit('Wheel build failed; Apple boards remain blocked')
artifact = B / 'apple-main-10523ede1'
artifact.mkdir(exist_ok=True)
remote_wheel = remote(hosts['m3ultra-b'], 'python3 -', '''import pathlib
w=list((pathlib.Path.home()/'mojolearn-wt/steward-m3ultra-b/python/dist').glob('*.whl'))
assert len(w)==1,w
print(w[0])
''').strip()
wheel = artifact / pathlib.Path(remote_wheel).name
if not wheel.exists():
    tmp = wheel.with_suffix('.partial')
    run(['scp', '-q', *ssh_opts, 'ec2-user@' + hosts['m3ultra-b'] + ':' + remote_wheel, str(tmp)])
    tmp.replace(wheel)
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
with zipfile.ZipFile(wheel) as z:
    assert z.read('mojolearn/identity_columns/COMMIT').decode().strip() == commit

receipts = {}
for name, host in hosts.items():
    check = remote(host, 'python3 -', 'import pathlib\np=pathlib.Path(' + repr(root) + ")/'installed-wheel.json'\nprint(p.read_text() if p.exists() else '{}')\n")
    receipt = json.loads(check)
    if receipt.get('sha256') != digest or receipt.get('source_commit') != commit:
        dest = root + '/wheel/' + wheel.name
        if name == 'm3ultra-b':
            remote(host, 'cp ' + shlex.quote(remote_wheel) + ' ' + shlex.quote(dest))
        else:
            run(['scp', '-q', *ssh_opts, str(wheel), 'ec2-user@' + host + ':' + dest])
        program = '''import hashlib,json,pathlib,subprocess,zipfile
r=pathlib.Path(ROOT);w=r/'wheel'/WHEEL
assert hashlib.sha256(w.read_bytes()).hexdigest()==DIGEST
with zipfile.ZipFile(w) as z:assert z.read('mojolearn/identity_columns/COMMIT').decode().strip()==COMMIT
py=r/'cache/venv/bin/python'
subprocess.run([str(py),'-m','pip','install','--no-deps','--force-reinstall',str(w)],check=True)
code="from mojolearn._backend import binding; import importlib.metadata as m; print(m.version('mojolearn')); "
code+="mods=[binding('_mojolearn_x_neighbors',s) for s in ['identical','fast']]; "
code+="assert all(b.x_neighbors_vendor()=='metal' and hasattr(b,'xn_lp_knn_graph') and hasattr(b,'xn_lp_knn_product') for b in mods); print('Installed Metal graph exports present in both tiers')"
subprocess.run([str(py),'-c',code],check=True,cwd='/tmp')
cfgp=r/'resume-config.json';cfg=json.loads(cfgp.read_text())
i=cfg['extra_args'].index('--mojolearn-wheel');cfg['extra_args'][i+1]=str(w)
cfgp.write_text(json.dumps(cfg,indent=2)+'\\n')
receipt={'source_commit':COMMIT,'sha256':DIGEST,'wheel':str(w),'published_to_pypi':False}
(r/'installed-wheel.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
        constants = 'ROOT=' + repr(root) + '\nWHEEL=' + repr(wheel.name) + '\nDIGEST=' + repr(digest) + '\nCOMMIT=' + repr(commit) + '\n'
        output = remote(host, 'python3 -', constants + program)
        (artifact / (name + '-install.log')).write_text(output)
        receipt = json.loads(output.strip().splitlines()[-1])
    # The installation receipt is written before unblocking, so a retry never
    # reinstalls a package while a board is using it.
    remote(host, 'python3 -', 'import pathlib\nr=pathlib.Path(' + repr(root) + ")\n(r/'DATA_READY').touch()\n(r/'blocked.json').unlink(missing_ok=True)\n")
    receipts[name] = receipt
    print(name, 'installed and unblocked', flush=True)
(B / 'apple-main-activated.json').write_text(json.dumps({'commit': commit, 'wheel_sha256': digest,
    'wheel': str(wheel), 'hosts': receipts, 'activated_utc': time.strftime('%FT%TZ',time.gmtime())},indent=2)+'\n')
