#!/usr/bin/env python3
"""Build the IDENTICAL grid's GPU bindings ONCE on a CPU-only build box, ship them to R2, and let the lq GPU boxes
install them instead of compiling (tools/six_lane_grid_lq.md, "Prebuilt bindings").

  plan        the deduplicated list of (binding, define set) artifacts the rendered lq lines need for one vendor
  build       cross-compile them in parallel with the SAME `mojo build` argv bindings/build_<x>.sh would use
  pack        one tar per vendor of the compiled artifacts + manifest.json + lookup.tsv (+ a .json sidecar)
  presign-put a presigned R2 PUT URL minted on the Mac (credential stays here) for the build box to upload to
  push        upload a tar to R2: with --put-url from any box, or with the Mac's ~/.mojolearn_r2 credential
  box-script  the shell script a GPU box runs to fetch, verify and unpack to /root/grid-prebuilt/<vendor>/

Dedup. Every lq line builds `build` + its BUILDS= scripts with the line's MOJOLEARN_BUILD_DEFINES. A define only
changes a binding's bits when its NAME is referenced by a file in that binding's Mojo import closure (the same
import graph the box uses to pick rebuilds: tools/hooks/no_host_routes.py Tree). So each (binding, defines that
reach it) is compiled once and serves every full define set that narrows to it; `lookup.tsv` maps
(binding, sha of the full set) -> artifact for tools/six_lane_grid_install_prebuilt.sh.

Fidelity. `build` never reconstructs a compile line. It runs the real bindings/build_<x>.sh with the box's
environment (MOJOLEARN_NUMERIC_MODE=identical, MOJOLEARN_TARGET_COLUMN=<vendor>, MOJOLEARN_GPU_ARCHS=sm_89|gfx942,
MOJOLEARN_BUILD_DEFINES=<effective set>) under a `pixi` shim that records the argv instead of compiling, then runs
that exact argv with the real compiler and `-o` pointed at the artifact. Host bindings (*_host) are not prebuilt:
races never load them (ID checks do, and ID lines never carry PREBUILT=).

Smoke on a Mac: the dry run emulates Linux (`uname` shim) so the recorded argv is the box's; the artifact is a
Mach-O and is marked box_usable=false (excluded from lookup.tsv and pack). Real artifacts come from a Linux
x86_64 box with the frozen tree's pixi env (`pixi run mojo`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
SCHEMA = 'mojolearn.six-lane-grid-prebuilt/1'
MODE = 'identical'
ARCH = {'nvidia': 'sm_89', 'amd': 'gfx942'}
GUARDS = 'core/six_lane_experiment_guards.mojo'  # names every define for conflict checks; never a bits input
DEFINES_ENV = 'MOJOLEARN_BUILD_DEFINES'
SECONDS_PER_BINDING = 60.0  # the brief's planning figure for one cross compile
INSTALL_SCRIPT = 'tools/six_lane_grid_install_prebuilt.sh'
DEFAULT_STORE = '/root/grid-prebuilt'
EMPTY_SHA = hashlib.sha256(b'').hexdigest()

# ------------------------------------------------------------------ small helpers


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def full_sha(defines):
    """sha256 of the sorted, deduplicated NAME[=V] entries, one per line (the install script's pipeline:
    `tr , '\\n' | sed '/^$/d' | LC_ALL=C sort -u | sha256sum`)."""
    return hashlib.sha256(''.join(d + '\n' for d in sorted({d for d in defines if d})).encode()).hexdigest()


def key_of(binding, effective):
    return hashlib.sha256(json.dumps([binding, MODE, sorted(effective)], separators=(',', ':')).encode()).hexdigest()[:16]


def script_binding(script):
    """build -> _mojolearn, build_x -> _mojolearn_x; None for host scripts and anything else."""
    if script == 'build':
        return '_mojolearn'
    m = re.fullmatch(r'build_(\w+)', script or '')
    if not m or m.group(1).endswith('_host') or m.group(1) == 'host_family':
        return None
    return '_mojolearn_' + m.group(1)


def define_name(d):
    return d.split('=', 1)[0]


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True).stdout.strip()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')


# ------------------------------------------------------------------ lq lines -> define sets


def parse_lines(text):
    """lq lines -> [{line, kind, defines, builds, prebuilt}] (only RACE and CMD lines; ID lines never prebuild)."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        toks = raw.split()
        if len(toks) < 4 or toks[:2] != ['lq', 'add'] or toks[3] not in ('RACE', 'CMD'):
            continue
        defines, builds, prebuilt = [], [], None
        for t in toks[4:]:
            if t.startswith(DEFINES_ENV + '='):
                defines = [d for d in t.split('=', 1)[1].split(',') if d]
            elif t.startswith('BUILDS='):
                builds = [b for b in t[len('BUILDS='):].split(',') if b]
            elif t.startswith('PREBUILT='):
                prebuilt = t[len('PREBUILT='):]
        if not builds:
            builds = ['build_x_linear']  # box_job.sh's default when BUILDS= is absent
        out.append(dict(line=n, kind=toks[3], defines=sorted(set(defines)), builds=builds, prebuilt=prebuilt))
    return out


def render_lines(plan_dir, vendor, branch):
    """The default render of the grid plan for the vendor (what the orchestrator queues), as text."""
    sys.path.insert(0, str(TOOLS))
    import six_lane_grid_lq as G  # metadata only
    lines, _ = G.render(Path(plan_dir), vendor, branch)
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------ reach: define -> bindings


def import_closures(root, bindings):
    """{binding: set(paths)} from the hook's import graph at HEAD (tools/hooks/no_host_routes.py Tree)."""
    sys.path.insert(0, str(Path(root) / 'tools' / 'hooks'))
    import no_host_routes as nhr  # noqa: E402
    cwd = os.getcwd()
    os.chdir(root)
    try:
        tree = nhr.Tree('HEAD')
    finally:
        os.chdir(cwd)
    out = {}
    for b in bindings:
        start = 'bindings/%s.mojo' % b
        seen, todo = set(), [start]
        while todo:
            p = todo.pop()
            if p in seen:
                continue
            seen.add(p)
            todo.extend(tree.edges.get(p, ()))
        out[b] = seen
    return out


def reach_map(root, bindings, names, closures=None):
    """{define NAME: sorted bindings whose import closure references it} (textual, GUARDS excluded)."""
    closures = closures or import_closures(root, bindings)
    texts = {}
    for paths in closures.values():
        for p in paths:
            if p not in texts and p != GUARDS:
                f = Path(root) / p
                texts[p] = f.read_text(errors='replace') if f.is_file() else ''
    pats = {n: re.compile(r'\b%s\b' % re.escape(n)) for n in names}
    out = {}
    for n in names:
        hit = set()
        for b, paths in closures.items():
            if any(pats[n].search(texts.get(p, '')) for p in paths):
                hit.add(b)
        out[n] = sorted(hit)
    return out


# ------------------------------------------------------------------ plan


def plan_artifacts(specs, reach, bindings_exist=None):
    """specs = parse_lines output; reach = {NAME: [bindings]}. Returns (artifacts, sets, skipped_scripts)."""
    sets, skipped = {}, {}
    for s in specs:
        fs = full_sha(s['defines'])
        e = sets.setdefault(fs, dict(sha=fs, defines=list(s['defines']), lines=0, scripts=set()))
        e['lines'] += 1
        for b in s['builds']:
            if script_binding(b) is None or (bindings_exist and not bindings_exist(b)):
                skipped[b] = skipped.get(b, 0) + 1
                continue
            e['scripts'].add(b)
    arts = {}
    pairs = 0
    for e in sets.values():
        for script in sorted(e['scripts']):
            binding = script_binding(script)
            pairs += 1
            eff = sorted(d for d in e['defines'] if binding in reach.get(define_name(d), []))
            k = key_of(binding, eff)
            a = arts.setdefault(k, dict(key=k, binding=binding, script=script, defines=eff, serves=[], lines=0))
            a['serves'].append(e['sha'])
            a['lines'] += e['lines']
    for a in arts.values():
        a['serves'] = sorted(set(a['serves']))
    ordered = sorted(arts.values(), key=lambda a: (-a['lines'], a['binding'], a['key']))
    for e in sets.values():
        e['scripts'] = sorted(e['scripts'])
    return ordered, sets, dict(pairs_binding_x_fullset=pairs, skipped_scripts=skipped)


def cmd_plan(args):
    vendor = args.vendor
    if args.lines:
        text = Path(args.lines).read_text()
        source = str(args.lines)
    else:
        text = render_lines(args.plan_dir, vendor, args.branch)
        source = 'render(%s, %s, %s)' % (args.plan_dir, vendor, args.branch)
    specs = parse_lines(text)
    if not specs:
        raise SystemExit('no RACE/CMD lq lines found in ' + source)
    root = Path(args.root or ROOT)
    scripts = sorted({b for s in specs for b in s['builds']})
    bindings = sorted({script_binding(b) for b in scripts if script_binding(b)})
    bindings = [b for b in bindings if (root / 'bindings' / (b + '.mojo')).is_file()]
    names = sorted({define_name(d) for s in specs for d in s['defines']})
    if args.reach_json:
        reach = json.loads(Path(args.reach_json).read_text())
    else:
        reach = reach_map(root, bindings, names)
    arts, sets, stats = plan_artifacts(specs, reach, lambda b: (root / 'bindings' / (b + '.sh')).is_file())
    box_builds = sum(len(s['builds']) for s in specs)
    unreferenced = sorted(n for n in names if not reach.get(n))
    est = {str(n): round(len(arts) * SECONDS_PER_BINDING / n / 60.0, 1) for n in (1, 4, 8, 16, 32)}
    doc = dict(schema=SCHEMA, kind='plan', vendor=vendor, arch=ARCH[vendor], mode=MODE,
               source_sha=git(root, 'rev-parse', 'HEAD'), plan_dir=str(args.plan_dir) if args.plan_dir else None,
               lines_source=source, lines=len(specs),
               counts=dict(lines=len(specs), binding_builds_on_box=box_builds, define_sets=len(sets),
                           artifacts_before_reach_dedup=stats['pairs_binding_x_fullset'], artifacts=len(arts),
                           bindings=len({a['binding'] for a in arts}), defines=len(names),
                           defines_unreferenced=len(unreferenced)),
               estimate_minutes_at_jobs=est, seconds_per_binding=SECONDS_PER_BINDING,
               skipped_scripts=stats['skipped_scripts'], defines_unreferenced=unreferenced,
               define_sets={k: dict(defines=v['defines'], lines=v['lines'], scripts=v['scripts']) for k, v in sets.items()},
               artifacts=arts)
    out = Path(args.out) / vendor
    write_json(out / 'plan.json', doc)
    write_json(out / 'reach.json', reach)
    print(json.dumps(dict(vendor=vendor, **doc['counts'], estimate_minutes_at_jobs=est, out=str(out / 'plan.json'))))
    return 0


# ------------------------------------------------------------------ build

PIXI_SHIM = '''#!/bin/sh
# dry-run shim for `pixi run mojo build ...`: record the argv, create the -o file with a marker, compile nothing
printf '%s\\n' "$@" > "$PREBUILD_SHIM_ARGV"
prev=
for a in "$@"; do
  if [ "$prev" = -o ]; then printf '%s' "$PREBUILD_SHIM_TOKEN" > "$a"; fi
  prev=$a
done
exit 0
'''
UNAME_SHIM = '''#!/bin/sh
# Linux x86_64 answers for a Mac smoke of the box's compile line
case "${1:-}" in -m|-p) echo x86_64;; -r) echo 6.0.0;; -a) echo "Linux buildbox 6.0.0 x86_64 GNU/Linux";; *) echo Linux;; esac
'''


def make_shims(d, emulate_linux):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for name, body in (('pixi', PIXI_SHIM),) + ((('uname', UNAME_SHIM),) if emulate_linux else ()):
        p = d / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return d


def box_env(base, vendor, defines, compile_jobs):
    env = {k: v for k, v in base.items() if not k.startswith('MOJOLEARN_') and k != 'MACOSX_DEPLOYMENT_TARGET'}
    env.update(MOJOLEARN_NUMERIC_MODE=MODE, MOJOLEARN_TARGET_COLUMN=vendor, MOJOLEARN_GPU_ARCHS=ARCH[vendor],
               MOJOLEARN_COMPILE_JOBS=str(compile_jobs), MOJOLEARN_BUILD_LOCK_HELD='1', MOJOLEARN_SKIP_BUILD_GATE='1',
               PYTHONUNBUFFERED='1')
    if defines:
        env[DEFINES_ENV] = ','.join(defines)
    return env


def dry_run(root, script, binding, vendor, defines, compile_jobs, shimdir, log):
    """Run bindings/<script>.sh under the pixi shim. Returns (recorded argv list, dest relpath or None, rc)."""
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix='prebuild-dry-') as td:
        argv_file = Path(td) / 'argv'
        token = 'PREBUILD-DRY-RUN-%s-%s' % (binding, os.getpid())
        env = box_env(os.environ, vendor, defines, compile_jobs)
        env['PATH'] = str(shimdir) + os.pathsep + env.get('PATH', '')
        env['PREBUILD_SHIM_ARGV'] = str(argv_file)
        env['PREBUILD_SHIM_TOKEN'] = token
        env['TMPDIR'] = td
        with Path(log).open('w') as f:
            f.write('env: %s\n' % ' '.join('%s=%s' % (k, shlex.quote(env[k])) for k in sorted(env) if k.startswith('MOJOLEARN_')))
            f.flush()
            try:
                rc = subprocess.run(['sh', 'bindings/%s.sh' % script], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=120).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        argv = argv_file.read_text().splitlines() if argv_file.is_file() else []
    dest = None
    pkg = root / 'python' / 'mojolearn'
    for p in sorted(pkg.rglob(binding + '.so')) if pkg.is_dir() else []:
        try:
            if p.stat().st_size < 256 and p.read_text(errors='replace') == token:
                dest = dest or str(p.relative_to(root))
                p.unlink()
        except OSError:
            pass
    return argv, dest, rc


def compile_argv(recorded, compiler, out_path):
    """`pixi run mojo build ... -o X` as the shim saw it (without `pixi`) -> `<compiler> build ... -o out_path`."""
    toks = list(recorded)
    if toks[:2] == ['run', 'mojo']:
        toks = toks[2:]
    if not toks or toks[0] != 'build':
        raise ValueError('recorded argv is not a mojo build line: %r' % (recorded[:4],))
    if '-o' not in toks:
        raise ValueError('recorded argv has no -o')
    i = toks.index('-o')
    toks[i + 1] = str(out_path)
    return list(compiler) + toks


def closure_sha(root, paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        f = Path(root) / p
        h.update(p.encode() + b'\0' + (sha_file(f).encode() if f.is_file() else b'missing') + b'\n')
    return h.hexdigest()


def build_one(a, ctx):
    """One artifact: dry run (serialized), compile, receipt. Returns the receipt."""
    root, out = ctx['root'], ctx['out']
    d = out / a['binding'] / a['key']
    d.mkdir(parents=True, exist_ok=True)
    so = d / (a['binding'] + '.so')
    receipt = d / 'receipt.json'
    if receipt.is_file():
        old = json.loads(receipt.read_text())
        if (old.get('status') == 'COMPILED' and so.is_file() and old.get('artifact_sha256') == sha_file(so)
                and old.get('source_sha') == ctx['source_sha'] and old.get('compiler_version') == ctx['compiler_version']):
            old['serves'] = sorted(set(old.get('serves', [])) | set(a['serves']))
            old['resumed'] = True
            write_json(receipt, old)
            return old
    t0 = time.time()
    with ctx['dry_lock']:
        argv_rec, dest, dry_rc = dry_run(root, a['script'], a['binding'], ctx['vendor'], a['defines'], ctx['compile_jobs'],
                                         ctx['shimdir'], d / 'dry-run.log')
    rec = dict(schema=SCHEMA, kind='receipt', key=a['key'], vendor=ctx['vendor'], arch=ARCH[ctx['vendor']], mode=MODE,
               binding=a['binding'], script='bindings/%s.sh' % a['script'], defines=a['defines'], serves=a['serves'],
               source_sha=ctx['source_sha'], source_dirty=ctx['dirty'],
               script_sha256=sha_file(root / 'bindings' / (a['script'] + '.sh')),
               build_defines_sha256=sha_file(root / 'bindings' / 'build_defines.sh'),
               closure_sha256=closure_sha(root, ctx['closures'].get(a['binding'], ())),
               closure_files=len(ctx['closures'].get(a['binding'], ())),
               compiler=' '.join(ctx['compiler']), compiler_version=ctx['compiler_version'],
               host=ctx['host'], box_usable=ctx['host']['box_usable'],
               dest=dest or 'python/mojolearn/identical/%s.so' % a['binding'], dest_source='dry-run' if dest else 'convention',
               argv_recorded=(['pixi'] + argv_rec) if argv_rec else [], dry_run_rc=dry_rc,
               artifact=str(so.relative_to(out)), log=str((d / 'build.log').relative_to(out)))
    if not argv_rec:
        rec.update(status='DRYRUN_FAILED', returncode=dry_rc, artifact_sha256=None, seconds=round(time.time() - t0, 1))
        write_json(receipt, rec)
        return rec
    tmp = d / (a['binding'] + '.so.tmp')
    try:
        argv = compile_argv(argv_rec, ctx['compiler'], tmp)
    except ValueError as exc:
        rec.update(status='DRYRUN_FAILED', returncode=dry_rc, error=str(exc), artifact_sha256=None,
                   seconds=round(time.time() - t0, 1))
        write_json(receipt, rec)
        return rec
    rec['argv'] = argv
    env = {k: v for k, v in os.environ.items() if not k.startswith('MOJOLEARN_') and k != 'MACOSX_DEPLOYMENT_TARGET'}
    with (d / 'build.log').open('w') as f:
        f.write('+ ' + ' '.join(shlex.quote(x) for x in argv) + '\n')
        f.flush()
        try:
            rc = subprocess.run(argv, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=ctx['timeout']).returncode
        except subprocess.TimeoutExpired:
            rc = 124
            f.write('\n+ timeout after %ss\n' % ctx['timeout'])
    if rc == 0 and tmp.is_file():
        tmp.replace(so)
        rec.update(status='COMPILED', returncode=0, artifact_sha256=sha_file(so), artifact_bytes=so.stat().st_size)
    else:
        if tmp.exists():
            tmp.unlink()
        rec.update(status='FAILED', returncode=rc, artifact_sha256=None)
    rec['seconds'] = round(time.time() - t0, 1)
    write_json(receipt, rec)
    return rec


def write_manifest(out, vendor, receipts, ctx_info):
    usable = [r for r in receipts if r.get('status') == 'COMPILED' and r.get('box_usable')]
    counts = dict(planned=len(receipts), compiled=sum(1 for r in receipts if r.get('status') == 'COMPILED'),
                  failed=sum(1 for r in receipts if r.get('status') == 'FAILED'),
                  dryrun_failed=sum(1 for r in receipts if r.get('status') == 'DRYRUN_FAILED'),
                  resumed=sum(1 for r in receipts if r.get('resumed')),
                  box_usable=len(usable), define_sets_served=len({s for r in usable for s in r['serves']}))
    man = dict(schema=SCHEMA, kind='manifest', vendor=vendor, arch=ARCH[vendor], mode=MODE, counts=counts, **ctx_info,
               artifacts=[{k: r.get(k) for k in ('key', 'binding', 'script', 'defines', 'serves', 'status', 'artifact',
                                                  'artifact_sha256', 'artifact_bytes', 'dest', 'box_usable', 'seconds',
                                                  'returncode')} for r in receipts])
    write_json(out / 'manifest.json', man)
    rows = ['# schema=%s vendor=%s arch=%s mode=%s source_sha=%s compiler_version=%s' % (
        SCHEMA, vendor, ARCH[vendor], MODE, ctx_info['source_sha'], ctx_info['compiler_version'].replace(' ', '_'))]
    rows.append('# binding\tfull_defines_sha256\tartifact\tartifact_sha256\tdest')
    for r in sorted(usable, key=lambda r: (r['binding'], r['key'])):
        for s in r['serves']:
            rows.append('\t'.join([r['binding'], s, r['artifact'], r['artifact_sha256'], r['dest']]))
    (out / 'lookup.tsv').write_text('\n'.join(rows) + '\n')
    return man


def host_info(emulate_linux, force_usable=False):
    system, machine = platform.system(), platform.machine()
    return dict(system=system, machine=machine, node=platform.node(), emulated_linux=bool(emulate_linux),
                box_usable=(system == 'Linux' and machine == 'x86_64') or bool(force_usable), box_usable_forced=bool(force_usable))


def cmd_build(args):
    out = Path(args.out) / args.vendor
    plan_path = out / 'plan.json'
    if not plan_path.is_file():
        raise SystemExit('no %s: run plan first' % plan_path)
    plan = json.loads(plan_path.read_text())
    root = Path(args.root or ROOT)
    head = git(root, 'rev-parse', 'HEAD')
    dirty = bool(git(root, 'status', '--porcelain', '--untracked-files=no'))
    if plan['source_sha'] != head:
        raise SystemExit('plan is for %s, tree is at %s: re-run plan' % (plan['source_sha'][:12], head[:12]))
    if dirty and not args.allow_dirty:
        raise SystemExit('tree has uncommitted changes; the box builds origin/<branch>. Commit, or --allow-dirty for a smoke')
    compiler = shlex.split(args.compiler)
    vp = subprocess.run(compiler + ['--version'], cwd=root, capture_output=True, text=True)
    if vp.returncode != 0:
        raise SystemExit('compiler probe failed: %s' % (vp.stderr or vp.stdout).strip()[:300])
    version = ' '.join((vp.stdout or vp.stderr).split())
    emulate = args.emulate_linux if args.emulate_linux is not None else platform.system() != 'Linux'
    host = host_info(emulate, args.force_box_usable)
    shimdir = make_shims(out / '.shim', emulate)
    arts = plan['artifacts']
    if args.binding:
        arts = [a for a in arts if a['binding'] in args.binding]
    if args.limit:
        arts = arts[:args.limit]
    bindings = sorted({a['binding'] for a in arts})
    closures = import_closures(root, bindings) if not args.no_closure else {}
    ctx = dict(root=root, out=out, vendor=args.vendor, source_sha=head, dirty=dirty, compiler=compiler,
               compiler_version=version, host=host, shimdir=shimdir, compile_jobs=args.compile_jobs,
               timeout=args.timeout, dry_lock=threading.Lock(), closures=closures)
    info = dict(source_sha=head, source_dirty=dirty, compiler=' '.join(compiler), compiler_version=version, host=host,
                plan_counts=plan['counts'], root=str(root), built_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    receipts = []
    lock = threading.Lock()

    def run(a):
        r = build_one(a, ctx)
        with lock:
            receipts.append(r)
            print(json.dumps(dict(key=r['key'], binding=r['binding'], status=r.get('status'), seconds=r.get('seconds'),
                                  defines=len(r['defines']))), flush=True)
            if len(receipts) % 10 == 0:
                write_manifest(out, args.vendor, receipts, info)
        return r

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        list(ex.map(run, arts))
    man = write_manifest(out, args.vendor, receipts, info)
    print(json.dumps(dict(vendor=args.vendor, **man['counts'], manifest=str(out / 'manifest.json'), host=host)))
    return 1 if man['counts']['failed'] or man['counts']['dryrun_failed'] else 0


# ------------------------------------------------------------------ pack / push / box-script


def cmd_pack(args):
    out = Path(args.out)
    vendors = [args.vendor] if args.vendor else [v for v in ARCH if (out / v / 'manifest.json').is_file()]
    if not vendors:
        raise SystemExit('no <out>/<vendor>/manifest.json under ' + str(out))
    for vendor in vendors:
        vd = out / vendor
        man = json.loads((vd / 'manifest.json').read_text())
        if man['counts']['box_usable'] == 0:
            print(json.dumps(dict(vendor=vendor, packed=0, note='no box-usable artifact (Mac smoke?)')))
            continue
        tar_path = out / ('grid-prebuilt-%s-%s.tar.gz' % (vendor, man['source_sha'][:12]))
        n = 0
        with tarfile.open(tar_path, 'w:gz') as tf:
            for name in ('manifest.json', 'lookup.tsv'):
                tf.add(vd / name, arcname='%s/%s' % (vendor, name))
            for a in man['artifacts']:
                if a['status'] != 'COMPILED' or not a['box_usable']:
                    continue
                ad = (vd / a['artifact']).parent
                for f in ('receipt.json', Path(a['artifact']).name):
                    tf.add(ad / f, arcname='%s/%s' % (vendor, (ad / f).relative_to(vd)))
                n += 1
        sha = sha_file(tar_path)
        side = dict(schema=SCHEMA, kind='tar', vendor=vendor, tar=str(tar_path), tar_name=tar_path.name, sha256=sha,
                    bytes=tar_path.stat().st_size, source_sha=man['source_sha'], compiler_version=man['compiler_version'],
                    artifacts=n, lookup_rows=sum(1 for ln in (vd / 'lookup.tsv').read_text().splitlines() if not ln.startswith('#')),
                    key='grid-prebuilt/%s/%s-%s.tar.gz' % (man['source_sha'], vendor, sha[:12]))
        write_json(Path(str(tar_path) + '.json'), side)
        print(json.dumps(dict(vendor=vendor, packed=n, tar=str(tar_path), sha256=sha, bytes=side['bytes'], key=side['key'],
                              sidecar=str(tar_path) + '.json')))
    return 0


def r2_creds(path):
    """The four R2_* values of ~/.mojolearn_r2 (a sh file), read through sh so quoting matches dataset_store.sh."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise SystemExit('no %s (the Mac keeps the R2 credential; a box uses --put-url)' % p)
    r = subprocess.run(['sh', '-c', '. "$1"; printf "%s\\n" "$R2_ACCOUNT_ID" "$R2_ACCESS_KEY_ID" "$R2_SECRET_ACCESS_KEY" "$R2_BUCKET"',
                        'sh', str(p)], capture_output=True, text=True)
    vals = r.stdout.splitlines()
    if r.returncode or len(vals) != 4 or not all(vals):
        raise SystemExit('%s is missing one of R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET' % p)
    return dict(zip(('account', 'key_id', 'secret', 'bucket'), vals))


def cmd_push(args):
    side_path = Path(args.sidecar)
    side = json.loads(side_path.read_text())
    tar_path = Path(side['tar']) if Path(side['tar']).is_file() else side_path.with_name(side['tar_name'])
    if not tar_path.is_file():
        raise SystemExit('tar not found: %s' % tar_path)
    if sha_file(tar_path) != side['sha256']:
        raise SystemExit('tar sha256 does not match its sidecar: %s' % tar_path)
    key = args.key or side['key']
    if args.put_url:
        # a presigned PUT minted on the Mac (presign-put); the box never sees a credential
        r = subprocess.run(['curl', '-sS', '-f', '--retry', '8', '--retry-delay', '5', '--retry-all-errors', '-T', str(tar_path),
                            args.put_url], capture_output=True, text=True)
        how = 'presigned PUT'
    else:
        c = r2_creds(args.creds)
        env = dict(os.environ, AWS_ACCESS_KEY_ID=c['key_id'], AWS_SECRET_ACCESS_KEY=c['secret'], AWS_DEFAULT_REGION='auto')
        r = subprocess.run(['aws', 's3', 'cp', str(tar_path), 's3://%s/%s' % (c['bucket'], key), '--endpoint-url',
                            'https://%s.r2.cloudflarestorage.com' % c['account'], '--only-show-errors'], env=env,
                           capture_output=True, text=True)
        how = 'aws s3 cp (Mac credential)'
    if r.returncode:
        raise SystemExit('push FAILED (%s): %s' % (how, (r.stderr or r.stdout).strip()[:400]))
    side.update(key=key, pushed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pushed_via=how)
    write_json(side_path, side)
    print(json.dumps(dict(pushed=str(tar_path), key=key, via=how, sha256=side['sha256'], bytes=side['bytes'])))
    return 0


def cmd_presign_put(args):
    r = subprocess.run(['sh', str(ROOT / 'tools' / 'dataset_store.sh'), 'presign-put', args.key, str(args.secs)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit((r.stderr or r.stdout).strip()[:400])
    print(r.stdout.strip())
    return 0


def presign_get(key, secs):
    r = subprocess.run(['sh', str(ROOT / 'tools' / 'dataset_store.sh'), 'presign', key, str(secs)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit('presign failed: ' + (r.stderr or r.stdout).strip()[:400])
    return r.stdout.strip()


def box_script(vendor, side, url, dest=DEFAULT_STORE):
    tar = side['tar_name']
    return '''#!/bin/bash
# Run ON the {vendor} lq box, no credentials here: fetch, verify and unpack the prebuilt grid bindings.
# source {src} compiler {cv} artifacts {n} lookup rows {rows}
set -eu
D={dest}; mkdir -p "$D"; T="$D/{tar}"
curl -sS -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$T" '{url}'
echo "{sha}  $T" | sha256sum -c -
rm -rf "$D/{vendor}.new"; mkdir "$D/{vendor}.new"
tar xzf "$T" -C "$D/{vendor}.new" --strip-components 1
( cd "$D/{vendor}.new" && grep -v '^#' lookup.tsv | awk -F'\\t' '{{print $4"  "$3}}' | sort -u | sha256sum -c --quiet - )
rm -rf "$D/{vendor}"; mv "$D/{vendor}.new" "$D/{vendor}"; rm -f "$T"
echo "PREBUILT {vendor} ready at $D/{vendor}: $(grep -vc '^#' "$D/{vendor}/lookup.tsv") lookup rows, source {src}"
'''.format(vendor=vendor, src=side['source_sha'], cv=side.get('compiler_version', '?'), n=side['artifacts'],
           rows=side['lookup_rows'], dest=dest, tar=tar, url=url, sha=side['sha256'])


def cmd_box_script(args):
    side = json.loads(Path(args.sidecar).read_text())
    if side.get('vendor') != args.vendor:
        raise SystemExit('sidecar is for %s, not %s' % (side.get('vendor'), args.vendor))
    key = args.key or side.get('key')
    if not key:
        raise SystemExit('sidecar has no key: push first, or pass --key')
    url = args.url or presign_get(key, args.secs)
    sys.stdout.write(box_script(args.vendor, side, url, args.dest))
    return 0


# ------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    a = s.add_parser('plan', help='deduplicated (binding, define set) artifacts for one vendor')
    a.add_argument('--plan-dir', type=Path, help='grid plan dir (grid-plan.json + grid-matrix.json.gz): rendered in-process')
    a.add_argument('--lines', type=Path, help='rendered lq lines file instead of rendering (the exact queued lines)')
    a.add_argument('--vendor', choices=sorted(ARCH), required=True)
    a.add_argument('--branch', default='main')
    a.add_argument('--out', type=Path, required=True, help='store root: <out>/<vendor>/plan.json, reach.json')
    a.add_argument('--root', help='source tree (default: this checkout)')
    a.add_argument('--reach-json', help='precomputed {NAME: [bindings]} instead of the import-closure scan (tests)')
    b = s.add_parser('build', help='cross-compile the planned artifacts in parallel')
    b.add_argument('--vendor', choices=sorted(ARCH), required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--jobs', type=int, default=4, help='parallel compiles')
    b.add_argument('--compiler', default='pixi run mojo', help='compiler command (the tree\'s pixi mojo by default)')
    b.add_argument('--compile-jobs', type=int, default=1, help='MOJOLEARN_COMPILE_JOBS (-j) per compile')
    b.add_argument('--limit', type=int)
    b.add_argument('--binding', action='append', help='only these bindings (repeatable)')
    b.add_argument('--timeout', type=int, default=3600, help='seconds per compile')
    b.add_argument('--allow-dirty', action='store_true', help='smoke only: build from a tree with uncommitted changes')
    b.add_argument('--emulate-linux', dest='emulate_linux', action='store_true', default=None,
                   help='uname shim for the dry run (default on a non-Linux host)')
    b.add_argument('--no-emulate-linux', dest='emulate_linux', action='store_false')
    b.add_argument('--no-closure', action='store_true', help='skip the closure digest (tests)')
    b.add_argument('--force-box-usable', action='store_true', help='tests/fake compiler only: list non-Linux artifacts in lookup.tsv')
    b.add_argument('--root')
    c = s.add_parser('pack', help='tar per vendor + sidecar json')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--vendor', choices=sorted(ARCH))
    d = s.add_parser('push', help='upload a packed tar to R2')
    d.add_argument('--sidecar', required=True, help='the <tar>.json pack wrote')
    d.add_argument('--put-url', help='presigned PUT URL (from the Mac: presign-put); no credential needed')
    d.add_argument('--key', help='R2 key (default: the sidecar\'s)')
    d.add_argument('--creds', default='~/.mojolearn_r2')
    e = s.add_parser('presign-put', help='mint a presigned PUT URL on the Mac (tools/dataset_store.sh presign-put)')
    e.add_argument('--key', required=True)
    e.add_argument('--secs', type=int, default=86400)
    f = s.add_parser('box-script', help='print the fetch+verify+unpack script a box runs')
    f.add_argument('vendor', choices=sorted(ARCH))
    f.add_argument('sidecar', help='the pushed tar\'s .json sidecar')
    f.add_argument('--dest', default=DEFAULT_STORE)
    f.add_argument('--secs', type=int, default=86400, help='GET URL validity')
    f.add_argument('--key')
    f.add_argument('--url', help='use this GET URL instead of minting one (tests)')
    args = p.parse_args(argv)
    if args.cmd == 'plan' and not (args.plan_dir or args.lines):
        p.error('plan needs --plan-dir or --lines')
    return dict(plan=cmd_plan, build=cmd_build, pack=cmd_pack, push=cmd_push, **{'presign-put': cmd_presign_put,
                'box-script': cmd_box_script})[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
