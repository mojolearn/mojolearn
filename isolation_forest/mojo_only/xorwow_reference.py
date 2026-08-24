"""Integer reference for cuRAND XORWOW as instantiated by cuML isolation forest:
curand_init(seed, tree_id, 0, &state); curand(); curand_uniform().
Source: upstream/curand-headers/nvidia/curand/include/curand_kernel.h (+ curand_precalc.h, curand_uniform.h).
Checks: (1) computed A^(2^67 * 4^i) matrices equal precalc_xorwow_matrix[i] for all 32 i;
        (2) computed A^(4^i) matrices equal precalc_xorwow_offset_matrix[i] for all 32 i
            (DEVIATION 751: cuML always passes offset=0, so the offset table is built and
            uploaded but never stepped; without this check half of DEVIATION 683 is unverified);
        (3) emits reference streams, including two (seed, tree, OFFSET) triples that do
            step the offset table."""
import re, struct, sys
M32 = 0xFFFFFFFF
HDR = "/Users/andrewhendel/CascadeProjects/upstream/curand-headers/nvidia/curand/include/curand_precalc.h"

def load_table(name):
    txt = open(HDR).read()
    i = txt.index(f"unsigned int {name}[32][800] = {{")
    j = txt.index("};", i)
    body = txt[i:j]
    rows = re.findall(r"\{([^{}]*)\}", body)
    assert len(rows) == 32, len(rows)
    out = []
    for r in rows:
        vals = [int(v) for v in re.findall(r"(\d+)UL", r)]
        assert len(vals) == 800, len(vals)
        out.append(vals)
    return out

# --- xorwow step on v[5] (linear over GF(2)), d separate
def xorwow_step_v(v):
    t = (v[0] ^ (v[0] >> 2)) & M32
    v0, v1, v2, v3, v4 = v[1], v[2], v[3], v[4], ((v[4] ^ ((v[4] << 4) & M32)) ^ (t ^ ((t << 1) & M32))) & M32
    return [v0, v1, v2, v3, v4]

# matrix in cuRAND layout: matrix[n*(i*32+j)+k] = k-th word of image of basis vector (word i, bit j)
N = 5
def matvec(vec, mat):
    res = [0]*N
    for i in range(N):
        for j in range(32):
            if vec[i] & (1 << j):
                for k in range(N):
                    res[k] ^= mat[N*(i*32+j)+k]
    return res

def matmat(A, B):  # A := A*B  (cuRAND __curand_matmat semantics: row r of A -> matvec(rowA, B))
    out = [0]*(N*N*32)
    for r in range(N*32):
        row = A[r*N:(r+1)*N]
        res = matvec(row, B)
        out[r*N:(r+1)*N] = res
    return out

def step_matrix():
    mat = [0]*(N*N*32)
    for i in range(N):
        for j in range(32):
            e = [0]*N; e[i] = 1 << j
            img = xorwow_step_v(e)
            for k in range(N):
                mat[N*(i*32+j)+k] = img[k]
    return mat

def matpow2k(A, k):  # A^(2^k)
    R = A
    for _ in range(k):
        R = matmat(R, R)
    return R

def build_sequence_matrices():
    A = step_matrix()
    S = matpow2k(A, 67)           # A^(2^67)
    mats = []
    for i in range(32):
        mats.append(S)
        S = matpow2k(S, 2)        # ^4
    return mats

def build_offset_matrices():
    # precalc_xorwow_offset_matrix[i] = A^(4^i); _skipahead_inplace applies
    # matrix i once per base-4 digit i of the offset.
    P = step_matrix()
    mats = []
    for i in range(32):
        mats.append(P)
        P = matpow2k(P, 2)        # ^4
    return mats

def curand_init(seed, subsequence, mats, offset=0, off_mats=None):
    seed &= (1<<64)-1
    s0 = (seed & M32) ^ 0xaad26b49
    s1 = ((seed >> 32) & M32) ^ 0xf7dcefdd
    t0 = (1099087573 * s0) & M32
    t1 = (2591861531 * s1) & M32
    d = (6615241 + t1 + t0) & M32
    v = [(123456789 + t0) & M32, (362436069 ^ t0) & M32, (521288629 + t1) & M32, (88675123 ^ t1) & M32, (5783321 + t0) & M32]
    x = subsequence; m = 0
    while x:
        for _ in range(x & 3):
            v = matvec(v, mats[m])
        x >>= 2; m += 1
    # _skipahead_inplace: the offset table, then d += 362437 * (unsigned int)offset
    x = offset; m = 0
    while x:
        for _ in range(x & 3):
            v = matvec(v, off_mats[m])
        x >>= 2; m += 1
    d = (d + 362437 * (offset & M32)) & M32
    return [d] + v

def curand(st):  # st = [d, v0..v4]; returns uint32, mutates
    d, v = st[0], st[1:]
    t = (v[0] ^ (v[0] >> 2)) & M32
    nv4 = ((v[4] ^ ((v[4] << 4) & M32)) ^ (t ^ ((t << 1) & M32))) & M32
    st[1:] = [v[1], v[2], v[3], v[4], nv4]
    st[0] = (d + 362437) & M32
    return (nv4 + st[0]) & M32

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def curand_uniform_bits(u32):
    # x * 2^-32f + 2^-33f, single rounding (fma or not: product exact)
    import numpy as np
    xf = np.float32(u32)            # cvt.rn.f32.u32
    r = np.float32(float(xf) * 2.0**-32 + 2.0**-33)   # exact in double, one rounding to f32
    return struct.unpack('<I', struct.pack('<f', float(r)))[0]

if __name__ == "__main__":
    tbl = load_table("precalc_xorwow_matrix")
    mats = build_sequence_matrices()
    bad = [i for i in range(32) if mats[i] != tbl[i]]
    print("sequence matrices equal curand_precalc.h:", "ALL 32" if not bad else f"MISMATCH {bad}")
    assert not bad, bad
    otbl = load_table("precalc_xorwow_offset_matrix")
    omats = build_offset_matrices()
    obad = [i for i in range(32) if omats[i] != otbl[i]]
    print("offset matrices equal curand_precalc.h:", "ALL 32" if not obad else f"MISMATCH {obad}")
    assert not obad, obad
    # DEVIATION 751 reach: an offset of 0 must leave the state alone, and a
    # nonzero offset must move it (otherwise the offset arm is inert).
    assert curand_init(42, 0, mats, 0, omats) == curand_init(42, 0, mats)
    assert curand_init(42, 0, mats, 1, omats) != curand_init(42, 0, mats)
    # ...and skipping `k` is stepping `k` times.
    for k in (1, 2, 3, 4, 5, 17, 64):
        st_skip = curand_init(7, 1, mats, k, omats)
        st_step = curand_init(7, 1, mats)
        for _ in range(k):
            curand(st_step)
        assert st_skip == st_step, (k, st_skip, st_step)
    print("offset skipahead == stepping, k in {1,2,3,4,5,17,64}: OK")
    # a sanity check on the step matrix vs direct stepping
    import random
    random.seed(1)
    v = [random.getrandbits(32) for _ in range(5)]
    assert matvec(v, step_matrix()) == xorwow_step_v(v)
    # reference streams
    out = []
    for seed, tree in [(0,0),(42,0),(42,1),(42,7),(0xDEADBEEFCAFEF00D,3),(123456789,255)]:
        st = curand_init(seed, tree, mats)
        vals = [curand(st) for _ in range(1024)]
        out.append((seed, tree, st, vals))
    with open(sys.argv[1] if len(sys.argv) > 1 else "xorwow_ref.tsv", "w") as f:
        f.write("# seed\ttree\tidx\tu32\tuniform_f32_bits\n")
        for seed, tree, st, vals in out:
            for i, u in enumerate(vals):
                f.write(f"{seed}\t{tree}\t{i}\t{u}\t{curand_uniform_bits(u):08x}\n")
    for seed, tree, st, vals in out:
        print(f"seed={seed} tree={tree} first4={vals[:4]} uni0=0x{curand_uniform_bits(vals[0]):08x}")
    # DEVIATION 751: the offset arm's own reference file. cuML never reaches
    # it, so nothing else in this lane would ever tell us the offset table is
    # wrong; these 6 x 256 draws do.
    off_out = []
    for seed, tree, off in [(0,0,1),(42,0,3),(42,1,4),(42,7,64),(0xDEADBEEFCAFEF00D,3,4095),(123456789,255,1)]:
        st = curand_init(seed, tree, mats, off, omats)
        vals = [curand(st) for _ in range(256)]
        off_out.append((seed, tree, off, vals))
    off_path = sys.argv[2] if len(sys.argv) > 2 else "xorwow_offset_reference.tsv"
    with open(off_path, "w") as f:
        f.write("# seed\ttree\toffset\tidx\tu32\tuniform_f32_bits\n")
        for seed, tree, off, vals in off_out:
            for i, u in enumerate(vals):
                f.write(f"{seed}\t{tree}\t{off}\t{i}\t{u}\t{curand_uniform_bits(u):08x}\n")
    for seed, tree, off, vals in off_out:
        print(f"seed={seed} tree={tree} offset={off} first4={vals[:4]}")
