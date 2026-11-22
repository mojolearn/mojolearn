"""Negative controls for the rlpair part: each patch breaks ONE assertion's
input and the part must answer RLPAIR_MOVED naming that assertion's pair."""
import sys, os, numpy as np
sys.path.insert(0, os.environ["WT"] + "/tools")
import identity_break as ib
import mojolearn as ml

X, yc, yr = ib.fixture("base"); Xh = ib.heldout("base")
orig_decode, orig_train, orig_take, orig_cat = ib._rl_decode, ib._rl_train, ib._rl_take, ib._rl_cat

def flip(a, idx):
    """flip the lowest bit of a[idx] in place; a is the (contiguous) result array"""
    a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint8)[idx] ^= 1

def c_sampler_alone():
    def d(ml_, side, prompts):
        r = orig_decode(ml_, side, prompts)
        if prompts.shape[0] == 1:
            flip(r[1], (0, 5))
        return r
    ib._rl_decode = d
def c_trainer_alone():
    def t(ml_, tr, prompts, ids, bounds):
        r = orig_train(ml_, tr, prompts, ids, bounds)
        if bounds == [0, 1]:
            flip(r[2], (0, 7, 0))
        return r
    ib._rl_train = t
def c_trainer_split():
    def t(ml_, tr, prompts, ids, bounds):
        r = orig_train(ml_, tr, prompts, ids, bounds)
        if len(bounds) == 3:
            flip(r[2], (3, 2, 0))
        return r
    ib._rl_train = t
def c_second_sampler():
    def d(ml_, side, prompts):
        r = orig_decode(ml_, side, prompts)
        if side.name.startswith("threaded="):
            flip(r[2], (4, 11, 0))
        return r
    ib._rl_decode = d
depth = [0]
def c_join_order():
    # top level only: SambaState's cat recurses through the module function
    def cat(states):
        if depth[0]:
            return orig_cat(states)
        depth[0] += 1
        try:
            return orig_cat([states[1], states[0]] + list(states[2:])) if len(states) == 3 else orig_cat(states)
        finally:
            depth[0] -= 1
    ib._rl_cat = cat
def c_leave_order():
    def take(st, rows):
        rows = list(rows)
        if depth[0]:
            return orig_take(st, rows)
        depth[0] += 1
        try:
            if len(rows) == 4 and rows != [0, 1, 2, 3]:
                rows = rows[::-1]
            return orig_take(st, rows)
        finally:
            depth[0] -= 1
    ib._rl_take = take

controls = [("sampler B=1", c_sampler_alone, ["mamba2", "transformer", "samba", "byte-lm-host-infer"]),
            ("trainer B=1", c_trainer_alone, ["mamba2", "transformer", "samba", "byte-lm-host-infer"]),
            ("trainer split", c_trainer_split, ["mamba2", "transformer", "samba", "byte-lm-host-infer"]),
            ("sampler threaded=", c_second_sampler, ["byte-lm-host-infer"]),
            ("continuous", c_join_order, ["mamba2", "transformer", "samba", "byte-lm-host-infer"]),
            ("continuous", c_leave_order, ["mamba2", "transformer", "samba", "byte-lm-host-infer"])]
fits = {}
bad = 0
for want, patch, lanes in controls:
    ib._rl_decode, ib._rl_train, ib._rl_take, ib._rl_cat = orig_decode, orig_train, orig_take, orig_cat
    patch()
    for name in lanes:
        if name not in fits:
            fits[name] = ib.LANES[name](ml, X, yc, yr, Xh.copy())
        v, err, sides = ib._probe_rlpair(fits[name], name, ml, Xh.copy(), False)
        ok = v is not None and v.startswith("RLPAIR_MOVED:" + want)
        bad += not ok
        print(f"{'FAILS AS IT MUST' if ok else 'CONTROL BROKEN'} {patch.__name__} {name}: {v or err}")
ib._rl_decode, ib._rl_train, ib._rl_take, ib._rl_cat = orig_decode, orig_train, orig_take, orig_cat
for name in fits:
    v, err, _ = ib._probe_rlpair(fits[name], name, ml, Xh.copy(), False)
    print(f"unpatched {name}: {v or err}")
print("controls broken:", bad)
