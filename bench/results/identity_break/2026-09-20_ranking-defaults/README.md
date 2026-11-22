# Ranking stochastic defaults, 2026-09-20

Source: `d97c62ad4b3558cd0811e2ed91f0cd0df0fcc991`.

The previous refusal claimed that QueryRMSE, PairLogit and YetiRank require
whole-query bootstrap. The pinned CatBoost greedy-searcher dispatch says
otherwise: `weak_objective_impl.h:21-43` samples the target row mapping;
`querywise_targets_impl.h:162-188` forms grouped gradients first, then
multiplies both planes by the sampled row weights. The implementation
already had that operation on CPU and GPU; the inaccurate restrictions
prevented those paths from running.

Both bindings were rebuilt from the changed source. The CPU and Apple M4
columns cover all nine fixtures, all rows, one fit per cell, and the full
applicable part set. Every numeric field agrees exactly. The lane fits all
three ranking losses with their default Bayesian bootstrap and score
noise. Predictions and loss curves cover all three; inference, saved-model
reload and batch checks use the QueryRMSE model.

Validation also ran 70 CPU tests, including 12 new ranking bootstrap tests.
An additional 18-configuration CPU/Apple comparison matched prediction and
loss-curve hashes: the three losses under default, no bootstrap, Bayesian
temperatures zero and one, Bernoulli and Poisson. Temperature zero exactly
recovered the unbootstrapped model; nonzero temperature changed it. A
separate fixed-seed repeat in the tests reproduced each default model.

Reproduce the recorded lane with an appropriate built binding set:

```sh
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python python tools/identity_break.py \
  --lanes gbdt-ranking-defaults --repeats 1 --json ranking-defaults.json
```

NVIDIA/AMD and sabotage-build columns are not part of this evidence.
