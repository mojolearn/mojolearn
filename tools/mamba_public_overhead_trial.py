#!/usr/bin/env python3
"""Exact per-call versus resident Mamba decode trial on one R2 object."""
import argparse, hashlib, json, math, mmap, statistics, time
from pathlib import Path

RUNG = {
    "screen": dict(dm=128, prefill=32, tokens=16, grad_length=8),
    "qualification": dict(dm=1024, prefill=256, tokens=64, grad_length=32),
}


class Bytes:
    def __init__(self, path, offset=0):
        self.f = Path(path).open("rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.at = offset % len(self.mm)

    def array(self, shape, scale=.0625, add=0):
        import numpy as np
        n = int(np.prod(shape)); limit = len(self.mm) - n
        if limit <= 0: raise ValueError("dataset too small")
        self.at %= limit
        raw = np.frombuffer(self.mm, np.uint8, n, self.at)
        self.at = (self.at + n + 2654435761) % limit
        return np.ascontiguousarray(
            ((raw.astype(np.float32) - np.float32(127.5)) *
             np.float32(scale / 127.5) + np.float32(add)).reshape(shape))

    def close(self): self.mm.close(); self.f.close()


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays: h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def weights(family, dm, source):
    di = 2 * dm
    if family == "mamba1":
        r = (dm + 15) // 16
        shapes = {"norm.weight":(dm,), "in_proj.weight":(2*di,dm),
                  "conv1d.weight":(di,1,4), "conv1d.bias":(di,),
                  "x_proj.weight":(r+32,di), "dt_proj.weight":(di,r),
                  "dt_proj.bias":(di,), "A_log":(di,16), "D":(di,),
                  "out_proj.weight":(dm,di)}
        ones = {"norm.weight"}
    elif family == "mamba2":
        nh=di//64; cd=di+256
        shapes={"block_norm.weight":(dm,), "in_proj.weight":(2*di+256+nh,dm),
                "conv1d.weight":(cd,1,4), "conv1d.bias":(cd,),
                "dt_bias":(nh,), "A_log":(nh,), "D":(nh,),
                "norm.weight":(di,), "out_proj.weight":(dm,di)}
        ones={"block_norm.weight","norm.weight"}
    else:
        nh=di//64
        shapes={"block_norm.weight":(dm,), "in_proj.weight":(2*di+256+3*nh+32,dm),
                "dt_bias":(nh,), "B_norm.weight":(128,), "C_norm.weight":(128,),
                "B_bias":(nh,128), "C_bias":(nh,128), "D":(nh,),
                "out_proj.weight":(dm,di)}
        ones={"block_norm.weight","B_norm.weight","C_norm.weight"}
    return {n: source.array(s, add=1 if n in ones else 0) for n,s in shapes.items()}


def make_block(family, w):
    import mojolearn as ml
    cls={"mamba1":ml.Mamba1Block,"mamba2":ml.Mamba2Block,"mamba3":ml.Mamba3Block}[family]
    b=cls(w); b.numeric_mode="identical"; return b


def state_arrays(family, state):
    names={
        "mamba1":("conv_window","h"),
        "mamba2":("conv_window","h","buffer_xbc","buffer_dtraw"),
        "mamba3":("theta","h","buffer_qrot","buffer_krot","buffer_v",
                  "buffer_dt","buffer_sig","buffer_adt","pending_k","pending_v"),
    }[family]
    return [getattr(state,n) for n in names]


def reports(family, block):
    names={"mamba1":(),"mamba2":("h_last_",),
           "mamba3":("h_last_","k_last_","v_last_","theta_last_")}[family]
    return [getattr(block,n) for n in names]


def arm(family, block, x, cfg, resident, sabotage=False):
    import numpy as np
    state=block.allocate_state(1)
    block.forward(np.ascontiguousarray(x[:,:cfg["prefill"]]),state)
    out=[]; trajectory=[]; samples=[]
    session=block.decode_session(state) if resident else None
    try:
        for pos in range(cfg["tokens"]):
            ix=cfg["prefill"]+pos
            if resident and sabotage and pos == cfg["tokens"]//2: ix += 1
            token=np.ascontiguousarray(x[:,ix:ix+1])
            t=time.perf_counter_ns()
            y=session.step(token) if resident else block.step(token,state)
            samples.append((time.perf_counter_ns()-t)/1e9)
            y=np.asarray(y).copy(); out.append(y)
            # Full-state witnessing is deliberately outside the timed step.
            # It proves every transition, rather than only the final cache.
            if session is not None: session.sync_state()
            trajectory.append(dict(output=sha(y),reports=sha(*reports(family,block)),
                                   state=sha(*state_arrays(family,state))))
        if session is not None: session.sync_state()
    finally:
        if session is not None: session.close()
    arrays=state_arrays(family,state)
    return dict(samples=samples,median=statistics.median(samples),
                output=sha(*out),state=sha(*arrays),trajectory=trajectory,
                quality_bits=sha(np.asarray([sum(float((z*z).sum()) for z in out)],dtype=np.float64)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family",choices=("mamba1","mamba2","mamba3"),required=True)
    ap.add_argument("--rung",choices=tuple(RUNG),default="screen")
    ap.add_argument("--dataset",type=Path,required=True); ap.add_argument("--dataset-key",required=True)
    ap.add_argument("--dataset-sha256",required=True); ap.add_argument("--commit",required=True)
    ap.add_argument("--process",type=int,required=True); ap.add_argument("--target-column",choices=("nvidia","amd"),required=True)
    ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--sabotage",choices=("none","wrong-token"),default="none")
    a=ap.parse_args(); import numpy as np; import mojolearn as ml
    cfg=RUNG[a.rung]; source=Bytes(a.dataset,104729)
    w=weights(a.family,cfg["dm"],source); block=make_block(a.family,w)
    common=Bytes(a.dataset,4000000)
    # One extra token exists solely for the wrong-token sabotage.
    x=common.array((1,cfg["prefill"]+cfg["tokens"]+1,cfg["dm"]))
    gx=common.array((1,cfg["grad_length"],cfg["dm"]))
    gdy=common.array(gx.shape)
    before=block.backward(gx,gdy); grad_before=sha(*[before[k] for k in sorted(before)])
    rows=[]; reference=None
    for rnd in range(a.rounds+1):
        order=("percall","resident") if rnd%2==0 else ("resident","percall")
        row=dict(round=rnd,warmup=rnd==0,order=list(order))
        for name in order:
            rec=arm(a.family,block,x,cfg,name=="resident",
                    a.sabotage=="wrong-token" and name=="resident")
            row[name]=rec
            if reference is None and name=="percall": reference=rec
        clean=a.sabotage=="none"
        equal=all(row["percall"][k]==row["resident"][k]
                  for k in ("output","state","trajectory","quality_bits"))
        if clean and not equal: raise RuntimeError("resident bytes diverged")
        if not clean and equal: raise RuntimeError("wrong-token sabotage did not diverge")
        rows.append(row)
    after=block.backward(gx,gdy); grad_after=sha(*[after[k] for k in sorted(after)])
    if grad_before != grad_after: raise RuntimeError("session changed backward gradients")
    retained=rows[1:]
    med={name:statistics.median(r[name]["median"] for r in retained)
         for name in ("percall","resident")}
    from mojolearn import _backend
    binding=_backend.binding("_mojolearn_mamba","identical"); bp=Path(binding.__file__)
    root=Path(__file__).resolve().parents[1]
    result=dict(schema="mojolearn.mamba-public-overhead.v1",family=a.family,rung=a.rung,
        shape=cfg,commit=a.commit,process=a.process,target_column=a.target_column,
        native_vendor=str(binding.mamba_vendor()),native_numeric_mode=int(binding.mamba_numeric_mode()),
        dataset=dict(key=a.dataset_key,sha256=a.dataset_sha256,bytes=a.dataset.stat().st_size),
        binding=dict(path=str(bp),bytes=bp.stat().st_size,sha256=hashlib.sha256(bp.read_bytes()).hexdigest()),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
          ("tools/mamba_public_overhead_trial.py","python/mojolearn/_mamba_impl.py","bindings/_mojolearn_mamba.mojo")},
        rounds=a.rounds,warmup=1,sabotage=a.sabotage,rows=rows,medians=med,
        speedup=med["percall"]/med["resident"],gradient_hash=grad_before,
        input_hash=sha(x),gradient_input_hash=sha(gx,gdy),weights_hash=sha(*[w[k] for k in sorted(w)]),
        exact=(a.sabotage=="none"),quality="bitwise output identity; quality_bits hashes identical")
    if not all(math.isfinite(v) and v>0 for v in med.values()): raise RuntimeError("invalid timing")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+"\n")
    print(json.dumps(dict(status="DIVERGENT" if a.sabotage!="none" else "PASS",speedup=result["speedup"])))
    source.close(); common.close()


if __name__ == "__main__": main()
