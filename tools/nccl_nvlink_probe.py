#!/usr/bin/env python3
"""NCCL all-reduce probe. NEVER RUN -- see the wind-down note below.

WOUND DOWN 2026-09-12 before it ever executed. `--mode timing` asks what a
deterministic all-reduce COSTS, and that question was withdrawn: a cost for
determinism can only come from comparing a deterministic arm against OUR OWN
optimized non-deterministic arm of the same operation, and we have none for
all-reduce. Timed against NCCL, the ratio is the sum of determinism,
implementation quality and configuration, which this experiment cannot
separate -- so no number it prints may be reported as the price of
determinism, on any hardware. `--mode hashes` and `--mode oracle` ask about
NCCL's own behavior instead, which is falsifiable and needs no arm of ours.

The original intent is kept below for whoever revisits the timing question
with a real optimized arm of our own to compare against.

What does a deterministic all-reduce cost when the optimized path is NOT handicapped?

This is the NVLink rerun of tools/nccl_determinism_probe.py. The input
generation, the bit hashing, the order-sensitivity oracle and the naive
fixed-order all-reduce are that probe's and are reused unchanged, because they
were already right. What is new here is the arm set:

  A   nccl        NCCL's all-reduce with NCCL choosing algo/proto/channels.
  B   nccl        the same call in a process whose NCCL_ALGO, NCCL_PROTO,
                  NCCL_MIN_NCHANNELS, NCCL_MAX_NCHANNELS and rank-to-device
                  mapping are all pinned. B is arm A's code; the difference is
                  the environment, which NCCL reads when the communicator is
                  built, so B can only be a SEPARATE process -- it is never
                  interleaved with A and the leg runs the two back to back,
                  more than once, so drift is visible.
  C1  det_gather  gather every rank's buffer to rank 0, sum in rank order,
                  broadcast. The naive fixed-order reduction. At W ranks it
                  pushes 2(W-1) messages through ONE device's links.
  C2  det_rs_ag   fixed-order reduce-scatter + all-gather. Rank r owns segment
                  r of every chunk, sums the W contributions to that segment in
                  RANK ORDER, and the reduced segments are all-gathered back.

C2 is the point of the rerun. Its arithmetic is elementwise
((x0 + x1) + x2) + x3 -- exactly C1's expression, element for element -- so C2
MUST hash identical to C1, and the probe asserts it rather than hoping. But the
bytes move like a ring: every link carries 2(W-1)/W of the buffer instead of
one device carrying all of it. The two data-movement collectives it uses
(all_to_all_single, all_gather_into_tensor) perform NO arithmetic, so NCCL is
free to route them however it likes without touching a single bit of the
result. That is what makes a deterministic reduction cheap: pin the ORDER OF
THE ADDITIONS, not the path of the bytes.

Chunking is likewise bit-neutral: chunks partition the element axis, and each
element's summation order is the same in every partition. `--mode invariance`
witnesses that by hashing C2 at several chunk counts and comparing.

Sizes are given in BYTES PER RANK, not elements, so a float32 row and a
bfloat16 row at "8 MB" move the same number of bytes and are comparable.

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


def make_input(rank, n, dtype, seed):
    """Deterministic per-rank buffer with magnitudes spread over 24 decades.

    Built on the CPU from a seeded generator so the bytes are identical on
    every process restart, on any device, and for any rank count. The sum is
    dominated by cancellation between terms 24 decades apart, which is exactly
    the regime where the summation order decides the result. Unchanged from
    tools/nccl_determinism_probe.py.
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
              "NCCL_MIN_NCHANNELS", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
              "NCCL_NVLS_ENABLE", "CUDA_VISIBLE_DEVICES"):
        rep[k] = os.environ.get(k, "")
    return rep


# ---------------------------------------------------------------- arms

def gather_to_zero(src, gathered, world, rank):
    """Every rank's buffer on rank 0, by point-to-point send/recv.

    Deliberately not the gather collective: irecv/isend moves the same bytes
    with the transfers overlapped, so the fixed-order arm is not handicapped by
    a serialized gather. Unchanged from tools/nccl_determinism_probe.py.
    """
    if rank == 0:
        gathered[0].copy_(src)
        reqs = [dist.irecv(gathered[i], src=i) for i in range(1, world)]
        for q in reqs:
            q.wait()
    else:
        dist.isend(src, dst=0).wait()
    torch.cuda.synchronize()


def det_gather(src, buf, gathered, world, rank):
    """C1: gather to rank 0, sum in rank order, broadcast.

    The arithmetic happens on one device in one order that does not depend on
    topology, message size or NCCL's algorithm choice, so its bits are a
    function of the inputs alone. Unchanged from the earlier probe.
    """
    gather_to_zero(src, gathered, world, rank)
    if rank == 0:
        buf.copy_(gathered[0])
        for i in range(1, world):
            buf += gathered[i]
    dist.broadcast(buf, src=0)


def det_rs_ag(src, buf, world, rank, chunks, scratch):
    """C2: fixed-order reduce-scatter + all-gather.

    Per chunk: all_to_all_single hands rank r the r-th segment of every rank's
    chunk, laid out by SOURCE RANK, so summing that buffer front to back is
    summing in rank order. all_gather_into_tensor puts the reduced segments
    back. Both collectives are pure data movement, so the result is
    ((x0 + x1) + ... ) elementwise in rank order -- bit-identical to C1, and to
    itself at any chunk count, on any topology, at any message size.
    """
    n = src.numel()
    per = n // chunks
    seg = per // world
    for c in range(chunks):
        a = c * per
        piece = src[a:a + per]
        recv = scratch[:per]
        dist.all_to_all_single(recv, piece)
        acc = recv[0:seg].clone()
        for s in range(1, world):
            acc += recv[s * seg:(s + 1) * seg]
        dist.all_gather_into_tensor(buf[a:a + per], acc)


def c2_divisible(n, world, chunks):
    return chunks > 0 and n % chunks == 0 and (n // chunks) % world == 0


# ---------------------------------------------------------------- modes

def mode_hashes(args, dtype, n, tag):
    """Is the collective bit-reproducible inside one process? (settled finding,
    re-confirmed here with peer-to-peer ENABLED)."""
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


def mode_oracle(args, dtype, n, tag):
    """Order-sensitivity witness on the same buffers the collective reduces.

    If this reports 1 distinct hash the inputs are benign and every verdict
    from that configuration is void -- which is what happens at 2 ranks, where
    a + b == b + a and there is only one order. Unchanged from the earlier
    probe except that the fixed-order arms are also hashed here, so the oracle
    says which of its orders C1 and C2 implement.
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


def mode_invariance(args, dtype, n, tag):
    """Does C2 give the same bits as C1, and the same bits at every chunk count?

    This is the claim that makes C2 usable: a deterministic reduction whose
    result does not depend on how the bytes were moved or how the buffer was
    cut up. A single disagreement here would sink the arm.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    src = make_input(rank, n, dtype, args.seed).cuda()
    buf = torch.empty_like(src)
    gathered = [torch.empty_like(src) for _ in range(world)] if rank == 0 else None
    scratch = torch.empty(n, dtype=dtype, device="cuda")

    det_gather(src, buf, gathered, world, rank)
    torch.cuda.synchronize()
    h_c1 = bit_hash(buf)

    h_c2 = {}
    for chunks in [int(c) for c in args.c2_chunks.split(",")]:
        if not c2_divisible(n, world, chunks):
            continue
        buf.zero_()
        det_rs_ag(src, buf, world, rank, chunks, scratch)
        torch.cuda.synchronize()
        h_c2[str(chunks)] = bit_hash(buf)

    buf.copy_(src)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    h_nccl = bit_hash(buf)

    emit({
        "mode": "invariance", "tag": tag, "dtype": str(dtype).split(".")[-1],
        "elements": n, "bytes": n * src.element_size(),
        "c1_hash": h_c1, "c2_hash_by_chunks": h_c2, "nccl_hash": h_nccl,
        "c2_equals_c1": all(v == h_c1 for v in h_c2.values()) and bool(h_c2),
        "c2_chunk_invariant": len(set(h_c2.values())) == 1 and bool(h_c2),
        "nccl_equals_c1": h_nccl == h_c1,
        **env_report(),
    })


def mode_timing(args, dtype, n, tag):
    """The three arms, interleaved rep by rep rather than blocked.

    Blocking an arm lets a clock or power state drift onto one arm alone; a
    round-robin over the arms puts any drift on all of them. Arm B is absent
    here by construction -- it is arm `nccl` in a pinned process.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    src = make_input(rank, n, dtype, args.seed).cuda()
    buf = torch.empty_like(src)
    arms_wanted = [a for a in args.arms.split(",") if a]
    gathered = ([torch.empty_like(src) for _ in range(world)]
                if (rank == 0 and "c1" in arms_wanted) else None)
    scratch = (torch.empty(n, dtype=dtype, device="cuda")
               if "c2" in arms_wanted else None)
    chunks = int(args.c2_chunks.split(",")[0])
    if "c2" in arms_wanted and not c2_divisible(n, world, chunks):
        chunks = 1

    def nccl_arm():
        buf.copy_(src)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)

    def c1_arm():
        det_gather(src, buf, gathered, world, rank)

    def c2_arm():
        det_rs_ag(src, buf, world, rank, chunks, scratch)

    fns = {"nccl": nccl_arm, "c1": c1_arm, "c2": c2_arm}
    arms = [a for a in arms_wanted if a in fns]

    for a in arms:                          # warm up every arm before any timing
        for _ in range(3):
            fns[a]()
    torch.cuda.synchronize(); dist.barrier()

    times = {a: [] for a in arms}
    for _ in range(args.reps):
        for a in arms:                      # INTERLEAVED: one rep of each, round robin
            torch.cuda.synchronize(); dist.barrier()
            t0 = time.perf_counter()
            fns[a]()
            torch.cuda.synchronize(); dist.barrier()
            times[a].append((time.perf_counter() - t0) * 1e3)

    hashes = {}
    for a in arms:
        fns[a]()
        torch.cuda.synchronize()
        hashes[a] = bit_hash(buf)

    row = {
        "mode": "timing", "tag": tag, "dtype": str(dtype).split(".")[-1],
        "elements": n, "bytes": n * src.element_size(), "reps": args.reps,
        "c2_chunks": chunks, "arms": arms,
        "ms_median": {a: statistics.median(times[a]) for a in arms},
        "ms_min": {a: min(times[a]) for a in arms},
        "ms_p90": {a: sorted(times[a])[int(0.9 * (len(times[a]) - 1))] for a in arms},
        "hash": hashes,
        **env_report(),
    }
    if "nccl" in arms:
        base = row["ms_median"]["nccl"]
        row["over_nccl"] = {a: row["ms_median"][a] / base for a in arms}
        row["equals_nccl"] = {a: hashes[a] == hashes["nccl"] for a in arms}
    emit(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["hashes", "oracle", "timing", "invariance"])
    # BYTES per rank, so float32 and bfloat16 rows move the same bytes.
    ap.add_argument("--sizes-bytes", default="1024,1048576,8388608,67108864,268435456")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--arms", default="nccl,c1,c2")
    ap.add_argument("--c2-chunks", dest="c2_chunks", default="4,1,8")
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
            isize = torch.empty(0, dtype=dtype).element_size()
            for nbytes in [int(s) for s in args.sizes_bytes.split(",")]:
                n = nbytes // isize
                if args.mode == "hashes":
                    mode_hashes(args, dtype, n, args.tag)
                elif args.mode == "oracle":
                    mode_oracle(args, dtype, n, args.tag)
                elif args.mode == "invariance":
                    mode_invariance(args, dtype, n, args.tag)
                else:
                    mode_timing(args, dtype, n, args.tag)
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
