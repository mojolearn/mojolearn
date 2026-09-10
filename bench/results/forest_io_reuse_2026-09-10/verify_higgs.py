import pathlib,json,hashlib,numpy as np
root=pathlib.Path('/root/datasets/gbm-bench/higgs');src=root/'higgs_speed.npz.partial'
m=json.loads(pathlib.Path('/root/forest_out/higgs-cache-manifest.json').read_text())
with src.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==m['cache_sha256']
with np.load(src) as z:
 x=z['x'];y=z['y'];assert list(x.shape)==m['cropped_shape']
 for k,a in [('X_train',x[:1000000]),('y_train',y[:1000000]),('X_test',x[-500000:]),('y_test',y[-500000:])]:assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==m['sha256'][k]
src.rename(root/'higgs_speed.npz')
pathlib.Path('/root/forest_out/higgs-cache.ready').write_text('verified compressed file and original train/test byte hashes\n')
print('PASS HIGGS1M + original fixed500k tail: cache/data SHA256 match')
