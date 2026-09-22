"""E2 witness on the M4: set_lr equals a fresh open with that lr; a different lr moves the bits."""
import argparse, sys
sys.path.insert(0, "/Users/andrewhendel/mojolearn-wt/gpt3-tooling/tools")
import numpy as np
import mojolearn as ml
from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
from mojolearn.cross_vendor import trainer_state_hash
import par_lm_xvendor as recipe

args = argparse.Namespace(seed=20260921, batch=2, length=32, d_model=32, heads=4, kv=2, ff=64, layers=2, vocab=512, steps=4, shards=4)
shape, state0, ids = recipe.build_problem(ml, args)
assert not callable(getattr(ml._backend.binding("_mojolearn_byte_lm", "identical"), "byte_lm_parallel_set_lr", None)) is False
print("binding has byte_lm_parallel_set_lr:", callable(getattr(ml._backend.binding("_mojolearn_byte_lm", "identical"), "byte_lm_parallel_set_lr", None)))
shards = lambda s: [np.ascontiguousarray(ids[s, k]) for k in range(4)]

# A: lr c1 for step 1, then a NEW trainer opened with lr c2 for step 2
c1, c2 = 1e-3, 3e-4
with Par(state0, devices=(0,), logical_shards=4) as t:
    t.train_step(shards(0)); s1 = t.state_dict(); h1a = trainer_state_hash(t)
s1["config"] = dict(s1["config"], lr=c2)
with Par(s1, devices=(0,), logical_shards=4) as t:
    t.train_step(shards(1)); h2a = trainer_state_hash(t)
# B: one trainer, set_lr(c2) between the steps
with Par(state0, devices=(0,), logical_shards=4) as t:
    t.train_step(shards(0)); h1b = trainer_state_hash(t)
    got = t.set_lr(c2)
    t.train_step(shards(1)); h2b = trainer_state_hash(t); cfg_lr = t.state_dict()["config"]["lr"]
# C: one trainer, set_lr(c1) (unchanged) between the steps: must differ from B at step 2
with Par(state0, devices=(0,), logical_shards=4) as t:
    t.train_step(shards(0)); t.set_lr(c1); t.train_step(shards(1)); h2c = trainer_state_hash(t)
# D: export_raw hash equals state_dict hash
with Par(state0, devices=(0,), logical_shards=4) as t:
    t.train_step(shards(0)); from mojolearn.cross_vendor import state_hash
    hd_raw, hd_sd = state_hash(t.export_raw()), state_hash(t.state_dict())
ok = (h1a == h1b) and (h2a == h2b) and (h2b != h2c) and (hd_raw == hd_sd) and (cfg_lr == np.float32(c2))
print("step1 equal:", h1a == h1b, "| set_lr == reopen:", h2a == h2b, "| lr change moves bits:", h2b != h2c,
      "| raw hash == state_dict hash:", hd_raw == hd_sd, "| config lr follows:", cfg_lr == np.float32(c2))
try:
    with Par(state0, devices=(0,), logical_shards=4) as t:
        t.set_lr(0.0); print("REFUSAL MISSING: lr 0 accepted"); ok = False
except ValueError as e:
    print("lr 0 refused:", e)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
