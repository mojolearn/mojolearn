#!/usr/bin/env python3
"""Small repeated-step screen for Transformer and Mamba-1/2/3.

Inputs and initial weights are deterministic float32 transforms of one pinned
R2 object.  This is an iteration screen, not the 162M/L2048 promotion gate.
The public block backward APIs recompute forward and return gradients to the
host, so the recorded boundary says so explicitly.
"""
import argparse, hashlib, json, mmap, statistics, time
from pathlib import Path


SHAPE = dict(batch=1, length=128, d_model=128)


def digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode()); h.update(memoryview(arrays[name]).cast('B'))
    return h.hexdigest()


class Bytes:
    def __init__(self, path, offset):
        self.stream = Path(path).open('rb')
        self.mm = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.cursor = offset % len(self.mm)

    def f32(self, shape, scale=.0625, add=0.0):
        import numpy as np
        n = int(np.prod(shape)); limit = len(self.mm) - n
        if limit <= 0: raise ValueError('dataset is too small')
        self.cursor %= limit
        raw = np.frombuffer(self.mm, dtype=np.uint8, count=n, offset=self.cursor)
        self.cursor = (self.cursor + 2654435761) % limit
        return np.ascontiguousarray((raw.astype(np.float32) - np.float32(127.5)) *
                                    np.float32(scale / 127.5) + np.float32(add)).reshape(shape)

    def close(self): self.mm.close(); self.stream.close()


def make(family, source):
    import numpy as np
    b, l, dm = SHAPE.values(); di = 2 * dm
    if family == 'transformer':
        weights = {
            'input_layernorm.weight': source.f32((dm,), add=1),
            'post_attention_layernorm.weight': source.f32((dm,), add=1),
            'q_proj.weight': source.f32((dm, dm)), 'k_proj.weight': source.f32((dm//2, dm)),
            'v_proj.weight': source.f32((dm//2, dm)), 'o_proj.weight': source.f32((dm, dm)),
            'gate_proj.weight': source.f32((2*dm, dm)), 'up_proj.weight': source.f32((2*dm, dm)),
            'down_proj.weight': source.f32((dm, 2*dm)),
        }
        from mojolearn import TransformerBlock
        block = TransformerBlock(weights, n_heads=4, n_kv_heads=2, head_dim=32)
        block.numeric_mode = 'identical'
    elif family == 'mamba1':
        rank = (dm + 15)//16
        weights = {'norm.weight': source.f32((dm,), add=1),
                   'in_proj.weight': source.f32((2*di, dm)),
                   'conv1d.weight': source.f32((di, 1, 4)), 'conv1d.bias': source.f32((di,)),
                   'x_proj.weight': source.f32((rank+32, di)),
                   'dt_proj.weight': source.f32((di, rank)), 'dt_proj.bias': source.f32((di,)),
                   'A_log': source.f32((di, 16)), 'D': source.f32((di,)),
                   'out_proj.weight': source.f32((dm, di))}
        from mojolearn import Mamba1Block
        block = Mamba1Block(weights); block.numeric_mode = 'identical'
    elif family == 'mamba2':
        heads = di//64; conv = di+256
        weights = {'block_norm.weight': source.f32((dm,), add=1),
                   'in_proj.weight': source.f32((2*di+256+heads, dm)),
                   'conv1d.weight': source.f32((conv, 1, 4)), 'conv1d.bias': source.f32((conv,)),
                   'dt_bias': source.f32((heads,)), 'A_log': source.f32((heads,)),
                   'D': source.f32((heads,)), 'norm.weight': source.f32((di,), add=1),
                   'out_proj.weight': source.f32((dm, di))}
        from mojolearn import Mamba2Block
        block = Mamba2Block(weights); block.numeric_mode = 'identical'
    else:
        heads = di//64
        weights = {'block_norm.weight': source.f32((dm,), add=1),
                   'in_proj.weight': source.f32((2*di+256+3*heads+32, dm)),
                   'dt_bias': source.f32((heads,)),
                   'B_norm.weight': source.f32((128,), add=1),
                   'C_norm.weight': source.f32((128,), add=1),
                   'B_bias': source.f32((heads,128)), 'C_bias': source.f32((heads,128)),
                   'D': source.f32((heads,)), 'out_proj.weight': source.f32((dm,di))}
        from mojolearn import Mamba3Block
        block = Mamba3Block(weights); block.numeric_mode = 'identical'
    return block, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=('transformer','mamba1','mamba2','mamba3'), required=True)
    ap.add_argument('--dataset', type=Path, required=True); ap.add_argument('--dataset-key', required=True)
    ap.add_argument('--dataset-sha256', required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True); ap.add_argument('--process', type=int, required=True)
    ap.add_argument('--target-column', choices=('nvidia','amd'), required=True)
    ap.add_argument('--warmup', type=int, default=1); ap.add_argument('--samples', type=int, default=5)
    args = ap.parse_args()
    import numpy as np
    source = Bytes(args.dataset, {'transformer':0,'mamba1':104729,'mamba2':209759,'mamba3':314573}[args.family])
    block, weights = make(args.family, source)
    common = Bytes(args.dataset, 4000000)
    x = common.f32((SHAPE['batch'],SHAPE['length'],SHAPE['d_model']))
    dy = common.f32((SHAPE['batch'],SHAPE['length'],SHAPE['d_model']))
    initial = digest(weights); rows=[]; last_y=None; last_g=None
    for step in range(args.warmup + args.samples):
        total0=time.perf_counter_ns(); t=time.perf_counter_ns(); last_y=block.forward(x)
        forward=(time.perf_counter_ns()-t)/1e9
        t=time.perf_counter_ns(); last_g=block.backward(x,dy); backward=(time.perf_counter_ns()-t)/1e9
        t=time.perf_counter_ns()
        for name, value in weights.items():
            value -= np.float32(1e-5) * np.asarray(last_g[name])
        optimizer=(time.perf_counter_ns()-t)/1e9; total=(time.perf_counter_ns()-total0)/1e9
        arrays = {k:np.asarray(v) for k,v in last_g.items()}
        if (not np.isfinite(np.asarray(last_y)).all() or
                not all(np.isfinite(v).all() for v in arrays.values()) or
                not all(np.isfinite(v).all() for v in weights.values())):
            raise RuntimeError('nonfinite family screen result')
        rows.append(dict(step=step+1,retained=step>=args.warmup,
                         forward=forward,backward=backward,optimizer=optimizer,total=total,
                         hashes=dict(output=hashlib.sha256(np.asarray(last_y).tobytes()).hexdigest(),
                                     gradients=digest(arrays),weights=digest(weights))))
    from mojolearn import _backend
    binding_name = '_mojolearn_transformer' if args.family == 'transformer' else '_mojolearn_mamba'
    binding = _backend.binding(binding_name, 'identical'); binding_path = Path(binding.__file__)
    root = Path(__file__).resolve().parents[1]
    source_names = ['tools/neural_family_screen.py','python/mojolearn/_transformer_impl.py',
                    'bindings/_mojolearn_transformer.mojo'] if args.family == 'transformer' else [
                    'tools/neural_family_screen.py','python/mojolearn/_mamba_impl.py','bindings/_mojolearn_mamba.mojo']
    source_sha256 = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names}
    retained = [r for r in rows if r['retained']]
    native_vendor = str(binding.transformer_vendor() if args.family == 'transformer' else binding.mamba_vendor())
    native_numeric_mode = int(binding.transformer_numeric_mode() if args.family == 'transformer' else binding.mamba_numeric_mode())
    result=dict(schema='mojolearn.neural-family-screen.v1',family=args.family,shape=SHAPE,
                commit=args.commit,process=args.process,target_column=args.target_column,
                binding=dict(path=str(binding_path),bytes=binding_path.stat().st_size,
                             sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest()),
                native_vendor=native_vendor,native_numeric_mode=native_numeric_mode,
                source_sha256=source_sha256,
                dataset=dict(key=args.dataset_key,sha256=args.dataset_sha256,bytes=args.dataset.stat().st_size),
                schedule='raw R2 bytes deterministically mapped to finite fp32 inputs, cotangents, and weights',
                warmup=args.warmup,samples=args.samples,rows=rows,
                medians={k:statistics.median(r[k] for r in retained) for k in ('forward','backward','optimizer','total')},
                hashes=dict(initial_weights=initial,final_weights=digest(weights),
                            input=hashlib.sha256(x.tobytes()).hexdigest(),
                            cotangent=hashlib.sha256(dy.tobytes()).hexdigest(),
                            output=hashlib.sha256(np.asarray(last_y).tobytes()).hexdigest(),
                            gradients=digest({k:np.asarray(v) for k,v in last_g.items()})),
                component_scope=dict(
                    forward='public synchronous block forward; includes projection and attention/scan core plus binding transfers',
                    backward='public synchronous VJP; recomputes forward and returns every gradient to host',
                    optimizer='host fp32 SGD over returned gradients; diagnostic, because no public Mamba model trainer exists',
                    attention_or_scan_core=None,pack_unpack_or_state_copies=None,
                    unavailable_reason='public block ABI does not expose isolated core/copy timers; use family-specific native timer follow-up after this total-step screen'))
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    common.close(); source.close(); print(json.dumps(dict(family=args.family,medians=result['medians'],status='PASS')))


if __name__ == '__main__': main()
