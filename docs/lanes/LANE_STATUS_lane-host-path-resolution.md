# lane/host-path-resolution — ONE resolution path for every host binding

2026-09-16. Status: MERGED to main. No box rented, no Metal job, one core.

## The defect

Four functions answered "where does this host binding live" and only two of
them read `MOJOLEARN_HOST_DIR`:

| resolver | honored the override, before |
|---|---|
| `_backend.host_dir()` | yes (it IS the override) |
| `_backend.host_module_path(basename)` | yes |
| `_backend.load_host_module(basename)` | yes (through `host_module_path`) |
| `_backend.host_binding_path()` | NO — built from `_pkg_dir()` |
| `_backend.forest_host_binding_path()` | NO — built from `_pkg_dir()` |
| `_byte_lm_host.binary_path()` | NO — built from `__file__` |
| `_forest_host.binary_path()` | NO — built from `__file__` |

The two in `_backend.py` had no callers in the tree. The two that matter are
the byte LM and forest INFERENCE doors, which every public CPU inference
surface goes through (`LanguageModelInference`,
`LanguageModelHostTrainer`, `HostForest`, `HostGBDT`, and `_gbdt_host.py`
via `_forest_host._load`). The byte LM TRAINING adapter
(`_byte_lm_trainer_host.binding()`) goes through `load_host_module` and did
honor the override, which is what split a run in half.

Two symptoms, one cause:

1. No host set in the package (a fresh worktree): the override is ignored,
   the inference door looks in the empty package directory and REFUSES,
   while the training door opens.
2. A host set present in the package: the override is ignored SILENTLY and
   the package's own binding answers though the process was told to use
   another set. This is the hazard `tools/identity_break.py:6210` already
   guards by hand, after a forest sabotage column read IDENTICAL on a
   RunPod x86 pod (2026-09-15).

## The reproduction (before the fix)

A worktree with no `python/mojolearn/host/`, the override naming main's set:

    MOJOLEARN_HOST_DIR=<main>/python/mojolearn/host MOJOLEARN_NUMERIC_MODE=identical \
    python3 tools/identity_break.py --lanes byte-lm,byte-lm-host-infer,byte-lm-host-train \
      --fixtures base --repeats 1

    | byte-lm                | f9a8d760f503f795 |   <- train, STABLE
    | byte-lm infer          | REFUSED          |
    | byte-lm batch          | REFUSED          |
    | byte-lm-host-infer     | REFUSED          |
    | byte-lm-host-train     | REFUSED          |

Every refusal named `<worktree>/python/mojolearn/host/_mojolearn_byte_lm_host.so`
while `MOJOLEARN_HOST_DIR` named a directory that HAS that file.

## The fix

`host_module_path()` is THE one resolution path. The two `_backend` helpers
delegate to it; the two inference doors keep their own named-binary variable
(`MOJOLEARN_BYTE_LM_HOST_BINARY`, `MOJOLEARN_FOREST_HOST_BINARY`, which the
CPU identity gate points at one file) and fall back to it instead of
building a path themselves. Four files, ~40 lines including docstrings:

    python/mojolearn/_backend.py
    python/mojolearn/_byte_lm_host.py
    python/mojolearn/_forest_host.py
    python/mojolearn/tests/test_host_path_resolution.py   (new)

Not touched, and correct as they are: `packaging/linux/stage_libs.py`,
`packaging/linux/pack_wheel.py` and `python/setup.py` resolve
`<package>/host/` while BUILDING a wheel, where the package directory is the
right and only answer.

## The test, seen failing first

`python/mojolearn/tests/test_host_path_resolution.py`, five tests. Against
the UNFIXED code, 3 failed / 2 passed — the two that pass are the
no-override tests, which is the point:

    FAILED test_under_the_override_every_resolver_follows_it
    FAILED test_a_relative_override_is_made_absolute
    FAILED test_the_inference_door_opens_under_the_override

Against the fixed code, 5 passed. The last one is the end-to-end arm: in a
subprocess (the module caches its binding per process), with the override
naming a mirror of the real set, the byte LM inference door must load the
MIRROR's file and answer loss bits and a logits SHA-256. Unfixed it loads
the package's file, so the arm fails on a full checkout too, not only on a
worktree with nothing built.

## No recorded hash moved

19 cells over `byte-lm`, `byte-lm-host-infer`, `byte-lm-host-train` and
`byte-lm-host-infer-threaded`, base fixture, compared against the unfixed
baseline:

    fixed, no override: compared=19 moved=0 missing=0 new=0
    fixed, override:    compared=19 moved=0 missing=0 new=0

The override run now produces the SAME hashes as the package-directory run,
which is what the override was always supposed to mean.

`byte-lm*/rlpair` reads REFUSED in every arm, before and after, for an
unrelated reason: no host binding covers `_mojolearn_training`. That is the
CPU training gap, not this lane.

## Gates (green on this branch)

    python3 tools/docs_facts.py --check                    # 13 facts, 12 marked spans
    python3 packaging/wheel_ci.py pins .                   # 56 build scripts
    python3 packaging/wheel_ci.py inventory python/mojolearn   # 85 modules

## Resume commands

    WT=<a worktree of main>; MAIN=/Users/andrewhendel/CascadeProjects/mojolearn
    PY=$MAIN/.pixi/envs/test/bin/python3
    export MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1

    # the test (needs a built host set for its last arm; it skips without one)
    cd $WT/python && nice -n 19 $PY -m pytest -q mojolearn/tests/test_host_path_resolution.py

    # the lane cells, under an override that mirrors the real set
    MIRROR=/tmp/host-mirror; mkdir -p $MIRROR
    for f in $MAIN/python/mojolearn/host/*.so; do ln -sfn "$f" "$MIRROR/"; done
    cd $WT && env PYTHONPATH=$WT/python MOJOLEARN_HOST_DIR=$MIRROR nice -n 19 $PY \
      tools/identity_break.py --lanes byte-lm,byte-lm-host-infer,byte-lm-host-train,byte-lm-host-infer-threaded \
      --fixtures base --repeats 1 --json /tmp/override.json

Known-failing BEFORE this lane and still failing, on a worktree with NO GPU
binary set built (verified identical on a detached worktree of the parent
commit, so neither is this lane's): 13 tests across
`test_host_model_linear_kernel.py` and `test_host_model_svm.py` (families
with no host binding), and `packaging/linux/selector_test.py` (19 passed, 6
failed).
