#!/usr/bin/env python3
"""Per-module SASS proof of the IDENTICAL PTX .rn pass.
usage: boxcheck.py <box3 dir> <out dir> <ptxas>
For every distinct module: ptxas (fmad=true/false) on the original, fmad=true on
the patched; driver JIT of both when the device can run the module's target."""
import collections, ctypes, json, multiprocessing as mp, os, re, subprocess, sys, tempfile
B, OUT, PTXAS = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
op_re = re.compile(r'/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)')


def sass_of(cub):
    t = subprocess.run(['cuobjdump', '-sass', cub], capture_output=True, text=True, check=True).stdout
    out = []
    for l in t.split('\n'):
        if op_re.search(l):
            out.append(re.sub(r'\s*/\*.*?\*/\s*', ' ', l).strip())
        elif l.strip().startswith('Function :'):
            out.append(l.strip())
    return out


def arch_of(p):
    return re.search(r'^\.target\s+(\w+)', open(p).read(), re.M).group(1)


def ptxas(args):
    path, fm = args
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        cub = f.name
    try:
        subprocess.run([PTXAS, f'-arch={arch_of(path)}', f'--fmad={fm}', path, '-o', cub], check=True, capture_output=True)
        return sass_of(cub)
    finally:
        os.unlink(cub)


def ops(s):
    c = collections.Counter()
    for l in s:
        m = re.match(r'(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)', l)
        if m:
            c[m.group(1)] += 1
    return c




def jit(ptx):
    """The driver JIT, as the runtime does it: the PTX handed to cuModuleLoadData."""
    d = open(ptx, 'rb').read() + b'\0'
    mod = ctypes.c_void_p()
    rc = cu.cuModuleLoadData(ctypes.byref(mod), d)
    if rc != 0:
        return None
    cu.cuModuleUnload(mod)
    # the cubin itself, through the linker (same JIT) for disassembly
    arch = arch_of(ptx)
    tgt = 0x10000 + int(arch[3:5]) if arch.endswith('a') else int(arch[3:])
    opts = (ctypes.c_int * 1)(9); vals = (ctypes.c_void_p * 1)(tgt)
    st = ctypes.c_void_p(); cu.cuLinkCreate_v2(1, opts, vals, ctypes.byref(st))
    assert cu.cuLinkAddData_v2(st, 1, d, ctypes.c_size_t(len(d)), b'm', 0, None, None) == 0
    cub = ctypes.c_void_p(); sz = ctypes.c_size_t()
    assert cu.cuLinkComplete(st, ctypes.byref(cub), ctypes.byref(sz)) == 0
    with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as f:
        f.write(ctypes.string_at(cub, sz.value)); name = f.name
    cu.cuLinkDestroy(st)
    try:
        return sass_of(name)
    finally:
        os.unlink(name)


names = sorted(os.listdir(os.path.join(B, 'orig')))
jobs = []
for n in names:
    o, p = os.path.join(B, 'orig', n), os.path.join(B, 'patched', n)
    jobs += [(o, 'true'), (o, 'false'), (p, 'true')]
with mp.Pool(os.cpu_count()) as pool:
    res = pool.map(ptxas, jobs, chunksize=4)
cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0)
dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
maj, mnr = ctypes.c_int(), ctypes.c_int()
cu.cuDeviceGetAttribute(ctypes.byref(maj), 75, dev); cu.cuDeviceGetAttribute(ctypes.byref(mnr), 76, dev)
local = maj.value * 10 + mnr.value
ctx = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)
drv = ctypes.c_int(); cu.cuDriverGetVersion(ctypes.byref(drv))
rows = []
agg = collections.Counter()
for i, n in enumerate(names):
    oT, oF, pT = res[3 * i:3 * i + 3]
    o, p = os.path.join(B, 'orig', n), os.path.join(B, 'patched', n)
    arch = arch_of(o)
    r = {'mod': n, 'arch': arch, 'text_changed': open(o).read() != open(p).read(),
         'orig_fmad_true_eq_false': oT == oF, 'patched_fmad_true_eq_orig_fmad_false': pT == oF}
    if int(arch[3:5]) <= local:
        oJ, pJ = jit(o), jit(p)
        r['jit_loaded'] = oJ is not None and pJ is not None
        if r['jit_loaded']:
            r['jit_patched_eq_jit_orig'] = oJ == pJ
            if oJ != pJ:
                a, b = ops(oJ), ops(pJ)
                r['jit_opcode_delta'] = {k: b[k] - a[k] for k in set(a) | set(b) if a[k] != b[k]}
    rows.append(r)
    for k, v in r.items():
        if v is True:
            agg[k] += 1
    agg['modules'] += 1
    agg['arch_' + arch] += 1
json.dump({'local_sm': local, 'driver': drv.value, 'rows': rows, 'summary': dict(agg)}, open(os.path.join(OUT, 'boxcheck.json'), 'w'), indent=1)
print('SUMMARY local_sm=%d driver=%d' % (local, drv.value), json.dumps(dict(agg)))
bad = [r for r in rows if not r['patched_fmad_true_eq_orig_fmad_false']]
print('patched fmad=true != orig fmad=false:', len(bad), [r['mod'] for r in bad][:20])
d = collections.Counter()
for r in rows:
    for k, v in r.get('jit_opcode_delta', {}).items():
        d[k] += v
print('JIT opcode delta over modules whose JIT SASS changed:', dict(d),
      'modules:', sum(1 for r in rows if r.get('jit_patched_eq_jit_orig') is False))
