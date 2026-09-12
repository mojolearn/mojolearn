#!/usr/bin/env python3
"""Is NCCL's all-reduce bitwise reproducible?

One process per GPU, launched by torchrun. Every mode hashes the raw bytes of
the reduced buffer, so "the same" means bit-for-bit the same, not close.

The trap this probe is built to avoid: a buffer of similar-magnitude values
sums to the same float in every order, so a probe on benign data reports
"deterministic" no matter what the collective did. Inputs here are drawn with
exponents spread over 10^-12 .. 10^+12 and random signs, so the sum is
dominated by cancellation and the result changes when the summation order
changes. `--mode oracle` proves that on the very buffers the collective is
handed: it sums the same per-rank buffers on ONE device in several different
orders and reports how many distinct hashes come out. If the oracle reports 1
distinct hash, the inputs are benign and every determinism verdict from that
configuration is void.

Modes:
  hashes  run the same all-reduce many times in one process, hash every result
  oracle  order-sensitivity witness (single device, host-gathered buffers)
  timing  NCCL all-reduce vs a fixed-order gather/sum/broadcast all-reduce

Output is one JSON object per line on rank 0's stdout, prefixed with JSON| so
it survives NCCL's own logging.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# Same element size as the payload, so a reinterpret-cast to an integer type
# gives numpy the exact bits without ever converting the value.
BITS_AS = {torch.float32: torch.int32, torch.bfloat16: torch.int16}


def emit(obj):
    if dist.get_rank() == 0:
        sys.stdout.write("JSON|" + json.dumps(obj, sort_keys=True) + "\n")
        sys.stdout.flush()


def bit_hash(t):
    """sha256 over the raw bytes of a tensor. No value conversion anywhere."""
    b = t.detach().cpu().contiguous().flatten().view(BITS_AS[t.dtype]).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:32]


def gather_to_zero(src, gathered, world, rank):
    """Every rank's buffer on rank 0, by point-to-point send/recv.

    Deliberately not the gather collective: the NCCL backend's support for it
    varies by torch version, and irecv/isend moves the same bytes with the
    transfers overlapped, so the fixed-order arm is not handicapped by a
    serialized gather.
    """
    if rank == 0:
        gathered[0].copy_(src)
        reqs = [dist.irecv(gathered[i], src=i) for i in range(1, world)]
        for q in reqs:
            q.wait()
    else:
        dist.isend(src, dst=0).wait()
    torch.cuda.synchronize()


def make_input(rank, n, dtype, seed):
    """Deterministic per-rank buffer with magnitudes spread over 24 decades.

    Built on the CPU from a seeded generator so the bytes are identical on
    every process restart, on any device, and for any rank count.
    """
    g = torch.Generator(device="cpu").manual_seed(seed * 100003 + rank)
    mant = torch.rand(n, generator=g, dtype=torch.float32) * 2.0 - 1.0
    expo = torch.randint(-12, 13, (n,), generator=g, dtype=torch.int32).to(torch.float32)
    x = mant * torch.pow(torch.tensor(10.0, dtype=torch.float32), expo)
    return x.to(dtype)


def env_report():
    rep = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl_version": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "device_name": torch.cuda.get_device_name(0),
        "world_size": dist.get_world_size(),
    }
    for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_NTHREADS", "NCCL_MAX_NCHANNELS",
              "NCCL_MIN_NCHANNELS", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"):
        rep[k] = os.environ.get(k, "")
    return rep


def mode_hashes(args, dtype, n, tag):
    rank = dist.get_rank()
    src = make_input(rank, n, dtype, args.seed).cuda()
    in_hash = bit_hash(src)
    buf = torch.empty_like(src)
    hashes = []
    for _ in range(args.iters):
        buf.copy_(src)                      # same input bytes every iteration
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        hashes.append(bit_hash(buf))
    uniq = sorted(set(hashes))
    emit({
        "mode": "hashes", "tag": tag, "dtype": str(dtype).split(".")[-1],
        "elements": n, "bytes": n * src.element_size(), "iters": args.iters,
        "rank0_input_hash": in_hash, "result_hash": hashes[0],
        "distinct_result_hashes": len(uniq), "all_hashes": uniq[:8],
        **env_report(),
    })
    return hashes[0]


def mode_oracle(args, dtype, n, tag):
    """Order-sensitivity witness on the same buffers the collective reduces.

    Every rank's buffer is gathered to rank 0 and summed there, on one device,
    in several orders: rank order, reverse order, a rotation, and a pairwise
    tree. If those disagree bit-for-bit, the inputs expose non-associativity
    and a determinism verdict on them is meaningful.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    src = make_input(rank, n, dtype, args.seed).cuda()
    gathered = [torch.empty_like(src) for _ in range(world)] if rank == 0 else None
    gather_to_zero(src, gathered, world, rank)
    if rank == 0:
        orders = {
            "forward": list(range(world)),
            "reverse": list(range(world))[::-1],
            "rotate": list(range(1, world)) + [0],
        }
        res = {}
        for name, order in orders.items():
            acc = gathered[order[0]].clone()
            for i in order[1:]:
                acc += gathered[i]
            res[name] = bit_hash(acc)
        if world % 2 == 0:                  # pairwise tree, the shape NCCL-tree mimics
            lvl = [g.clone() for g in gathered]
            while len(lvl) > 1:
                lvl = [lvl[i] + lvl[i + 1] for i in range(0, len(lvl), 2)]
            res["tree"] = bit_hash(lvl[0])
        emit({
            "mode": "oracle", "tag": tag, "dtype": str(dtype).split(".")[-1],
            "elements": n, "orders": res, "distinct_order_hashes": len(set(res.values())),
            **env_report(),
        })
    dist.barrier()


def det_allreduce(src, buf, gathered, world, rank):
    """A fixed-order all-reduce: gather to rank 0, sum in rank order, broadcast.

    The arithmetic happens on one device in one order that does not depend on
    topology, message size or NCCL's algorithm choice, so its bits are a
    function of the inputs alone.
    """
    gather_to_zero(src, gathered, world, rank)
    if rank == 0:
        buf.copy_(gathered[0])
        for i in range(1, world):
            buf += gathered[i]
    dist.broadcast(buf, src=0)


def mode_timing(args, dtype, n, tag):
    rank, world = dist.get_rank(), dist.get_world_size()
    src = make_input(rank, n, dtype, args.seed).cuda()
    buf = torch.empty_like(src)
    gathered = [torch.empty_like(src) for _ in range(world)] if rank == 0 else None

    def timed(fn, reps):
        # Warm up, then one timed pass at a time. Nothing else runs on these
        # GPUs while this loop runs; the two arms never overlap.
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        out = []
        for _ in range(reps):
            torch.cuda.synchronize(); dist.barrier()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(); dist.barrier()
            out.append((time.perf_counter() - t0) * 1e3)
        return out

    def nccl_arm():
        buf.copy_(src)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    def det_arm():
        det_allreduce(src, buf, gathered, world, rank)

    t_nccl = timed(nccl_arm, args.reps)
    nccl_arm(); torch.cuda.synchronize(); h_nccl = bit_hash(buf)
    t_det = timed(det_arm, args.reps)
    det_arm(); torch.cuda.synchronize(); h_det = bit_hash(buf)
    emit({
        "mode": "timing", "tag": tag, "dtype": str(dtype).split(".")[-1],
        "elements": n, "bytes": n * src.element_size(), "reps": args.reps,
        "nccl_ms_median": statistics.median(t_nccl), "nccl_ms_min": min(t_nccl),
        "det_ms_median": statistics.median(t_det), "det_ms_min": min(t_det),
        "det_over_nccl": statistics.median(t_det) / statistics.median(t_nccl),
        "nccl_hash": h_nccl, "det_hash": h_det, "nccl_equals_det": h_nccl == h_det,
        **env_report(),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["hashes", "oracle", "timing"])
    ap.add_argument("--sizes", default="256,16384,262144,2097152,16777216,67108864")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        for dname in args.dtypes.split(","):
            dtype = DTYPES[dname]
            for n in [int(s) for s in args.sizes.split(",")]:
                if args.mode == "hashes":
                    mode_hashes(args, dtype, n, args.tag)
                elif args.mode == "oracle":
                    mode_oracle(args, dtype, n, args.tag)
                else:
                    mode_timing(args, dtype, n, args.tag)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
