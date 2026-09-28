#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A host-logic model of the Lossguide exact batch (trees-apple3,
`greedy_search_helper_depthwise.mojo`, `_lg_exact_plan` and the driver's
bookkeeping around it), in plain Python: no device, no library, one core,
seconds. It grows random trees whose gains are a function of the node's path
and asserts that the batched replay returns best-first's leaves, in
best-first's leaf-id order, across leaf budgets, depth bounds, widths, ties,
undefined splits, terminal leaves and min_split_gain. It is a model of the
logic, not a check of the compiled driver.

    python3 bench/speed/trees_apple3_lg_exact_model.py 3000
"""
import random, sys
UNKNOWN, DEFINED, NOSPLIT = 0, 1, 2
FMAX = 3.4e38

def plan(left, right, gain, state, max_leaves, msg, limit):
    out = dict(complete=False, blocked=False, final=[], expand=[], picks=0)
    sim = [0]; exact = True
    while True:
        if len(sim) >= max_leaves:
            if exact: out['complete'] = True
            break
        best = -1; bg = FMAX; unknown = False
        for i, n in enumerate(sim):
            if n < 0: continue
            if state[n] == UNKNOWN: unknown = True; continue
            if state[n] != DEFINED: continue
            if gain[n] < bg: bg = gain[n]; best = i
        if exact and unknown:
            out['blocked'] = True; break
        if best < 0:
            if exact: out['complete'] = True
            break
        if msg >= 0 and not (-bg > msg):
            if exact: out['complete'] = True
            break
        p = sim[best]
        if left[p] >= 0:
            sim[best] = left[p]; sim.append(right[p])
            if exact: out['picks'] += 1
        else:
            exact = False
            if len(out['expand']) >= limit: break
            out['expand'].append(p)
            sim[best] = -1; sim.append(-1)
    if out['complete']: out['final'] = list(sim)
    return out

class World:
    """Gains are a function of the node's PATH (a tuple of 0/1), so both
    growers see the same numbers whatever order they split in."""
    def __init__(self, seed, max_depth, ties, p_undef):
        self.seed, self.max_depth, self.ties, self.p_undef = seed, max_depth, ties, p_undef
    def gain(self, path):
        r = random.Random(hash((self.seed, path)))
        if r.random() < self.p_undef: return None
        g = -r.random() * (0.7 ** len(path)) + (0.2 if r.random() < 0.2 else 0.0) * r.random()
        if self.ties: g = round(g, 1)
        return g
    def terminal(self, path):
        r = random.Random(hash((self.seed, path, 't')))
        return len(path) >= self.max_depth or r.random() < 0.05

def best_first(w, M, msg):
    leaves = [()]  # paths by leaf id
    g = {(): w.gain(())}
    term = {(): False}
    while True:
        best = -1; bg = FMAX
        for i, p in enumerate(leaves):
            if term[p] or g[p] is None: continue
            if g[p] < bg: bg = g[p]; best = i
        if best < 0: break
        if msg >= 0 and not (-bg > msg): break
        p = leaves[best]
        l, r = p + (0,), p + (1,)
        leaves[best] = l; leaves.append(r)
        for c in (l, r):
            term[c] = w.terminal(c)
            g[c] = None if term[c] else w.gain(c)
        if len(leaves) >= M: break
    return leaves

def batched(w, M, msg, width, max_depth):
    phys = 1 << max_depth
    cap = M; bound = False
    if phys > M:
        cap = 2 * M
        if cap >= phys: cap = phys
        else: bound = True
    leaf_path = [()]           # device leaves
    leaf_node = [0]
    nleaf = [0]; left = [-1]; right = [-1]; gain = [FMAX]; state = [UNKNOWN]; npath = [()]
    term = [False]
    rounds = 0; wasted = 0
    while True:
        rounds += 1
        # score every nonterminal unscored leaf
        for lid, p in enumerate(leaf_path):
            n = leaf_node[lid]
            if state[n] == UNKNOWN and not term[lid]:
                gv = w.gain(p)
                if gv is None: state[n] = NOSPLIT
                else: state[n] = DEFINED; gain[n] = gv
        limit = min(width, cap - len(leaf_path))
        pl = plan(left, right, gain, state, M, msg, 1)
        assert not pl['blocked']
        if pl['complete']:
            final = pl['final']; break
        if bound:
            safe = cap + 2 - M - len(leaf_path) + pl['picks']
            limit = min(limit, safe)
        assert limit >= 1, (limit, len(leaf_path), cap)
        if limit > 1:
            pl = plan(left, right, gain, state, M, msg, limit)
        to_split = sorted(nleaf[n] for n in pl['expand'])
        count = len(leaf_path)
        for i, lid in enumerate(to_split):
            rid = count + i
            pn = leaf_node[lid]
            p = leaf_path[lid]
            npath[pn] = p
            cn = len(nleaf)
            left[pn] = cn; right[pn] = cn + 1
            nleaf += [lid, rid]
            left += [-1, -1]; right += [-1, -1]; gain += [FMAX, FMAX]; state += [UNKNOWN, UNKNOWN]
            npath += [None, None]
            leaf_node[lid] = cn; leaf_node.append(cn + 1)
            leaf_path[lid] = p + (0,); leaf_path.append(p + (1,))
            term[lid] = w.terminal(leaf_path[lid]); term.append(w.terminal(leaf_path[rid]))
        assert len(leaf_path) <= cap, (len(leaf_path), cap)
        for i, lid in enumerate(to_split):
            for l2 in (lid, count + i):
                if term[l2]: state[leaf_node[l2]] = NOSPLIT
        chk = plan(left, right, gain, state, M, msg, 0)
        if chk['complete']:
            final = chk['final']; break
    paths = []
    for n in final:
        paths.append(npath[n] if left[n] >= 0 else leaf_path[nleaf[n]])
    wasted = (len(leaf_path) - 1) - (len(final) - 1)
    return paths, rounds, wasted

def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    rnd = random.Random(5)
    tot_r = tot_bf = tot_w = 0
    for t in range(trials):
        M = rnd.choice([2, 3, 5, 8, 16, 31, 64, 100])
        md = rnd.choice([1, 2, 3, 4, 6, 8, 10, 12])
        width = rnd.choice([1, 2, 16, 32, 64])
        msg = rnd.choice([-1, -1, 0.0, 0.05])
        w = World(t, md, ties=rnd.random() < 0.4, p_undef=rnd.choice([0.0, 0.1]))
        a = best_first(w, M, msg)
        b, rounds, wasted = batched(w, M, msg, width, md)
        assert a == b, (t, M, md, width, msg, a, b)
        tot_r += rounds; tot_bf += len(a) - 1; tot_w += wasted
    print("trials", trials, "all equal (leaf paths in best-first leaf-id order)")
    print("best-first iterations", tot_bf, "batched rounds", tot_r, "leaves split ahead and folded back", tot_w)
main()
