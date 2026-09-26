"""On the box: rn wheel -> wheel whose IDENTICAL CUDA sets carry cubin.
usage: mkcubin.py <src.whl> <dst.whl> <ptxas> <report.json> [arch ...]"""
import shutil, base64, hashlib, json, os, sys, time, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubin_contract as cc
import ptx_contract as pc

src, dst, ptxas, report = sys.argv[1:5]
archs = sys.argv[5:] or ['sm_89', 'sm_90a']
zin = zipfile.ZipFile(src)
infos = {i.filename: i for i in zin.infolist()}
data = {n: zin.read(n) for n in infos}
dist = [n.split('/')[0] for n in data if n.endswith('.dist-info/RECORD')][0]
fatbinary = os.path.join(os.path.dirname(ptxas), 'fatbinary')
rep = {'ptxas': cc.tool_version(ptxas), 'fatbinary': cc.tool_version(fatbinary), 'ptxas_flags': cc.PTXAS_FLAGS, 'fatbinary_flags': cc.FATBIN_FLAGS, 'sets': {}}
t0 = time.time()
with tempfile.TemporaryDirectory() as t:
    for arch in archs:
        names = [n for n in data if pc.is_identical_cuda(n) and n.split('/')[2] == arch]
        paths = []
        for n in names:
            p = os.path.join(t, n.replace('/', '__'))
            open(p, 'wb').write(data[n])
            paths.append(p)
        rows = cc.patch_files(paths, ptxas, fatbinary, arch=arch, objdump=shutil.which('objdump'))
        for n, p in zip(names, paths):
            new = open(p, 'rb').read()
            assert len(new) == len(data[n])
            data[n] = new
        rep['sets'][arch] = {'rows': rows, 'files': len(names),
                             'modules': sum(r['modules'] for r in rows),
                             'in_place': sum(r['in_place'] for r in rows),
                             'moved': sum(r['moved'] for r in rows),
                             'unplaced': sum(len(r['unplaced']) for r in rows),
                             'ptx_bytes': sum(r['ptx_bytes'] for r in rows),
                             'image_bytes': sum(r['image_bytes'] for r in rows)}
        print(arch, {k: v for k, v in rep['sets'][arch].items() if k != 'rows'}, f'{time.time() - t0:.0f}s', flush=True)
changed = [n for n in data if pc.is_identical_cuda(n) and n.split('/')[2] in archs]
payload = json.loads(data[f'{dist}/LINUX_PAYLOAD.json'])
for name in changed:
    assert name in payload['extensions'], name
    payload['extensions'][name] = hashlib.sha256(data[name]).hexdigest()
data[f'{dist}/LINUX_PAYLOAD.json'] = (json.dumps(payload, sort_keys=True, indent=2) + '\n').encode()
rows = []
for name in data:
    if name == f'{dist}/RECORD':
        continue
    d = base64.urlsafe_b64encode(hashlib.sha256(data[name]).digest()).decode().rstrip('=')
    rows.append(f'{name},sha256={d},{len(data[name])}')
rows.append(f'{dist}/RECORD,,')
data[f'{dist}/RECORD'] = ('\n'.join(rows) + '\n').encode()
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in data:
        z.writestr(infos[name], data[name])
# sizes: compressed bytes per arch identical tier, before and after
def csize(whl, arch):
    return sum(i.compress_size for i in zipfile.ZipFile(whl).infolist()
               if pc.is_identical_cuda(i.filename) and i.filename.split('/')[2] == arch)
rep['wheel_bytes'] = {'before': os.path.getsize(src), 'after': os.path.getsize(dst)}
rep['identical_compressed'] = {a: {'before': csize(src, a), 'after': csize(dst, a)} for a in archs}
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile(dst).extractall(t)
    errors, arows = cc.audit_tree(t)
    rep['audit_errors'] = errors
    rep['audit_rows'] = arows
json.dump(rep, open(report, 'w'), indent=1)
print('wheel', rep['wheel_bytes'], 'identical compressed', rep['identical_compressed'])
print('audit errors', len(errors), errors[:5])
