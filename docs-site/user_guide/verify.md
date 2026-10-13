# Verify on your hardware

Each wheel ships the reference results recorded for its release and a verifier that checks your
machine against them. No repository, dataset, network or other machine-learning library is needed.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs. The verifier checks your machine against the same expected hashes on every vendor.

## Install verification support

```sh
python -m pip install "mojolearn[verify]"
```

The `verify` extra is the same library plus NumPy, which the verifier uses to build its fixtures and
inspect arrays. It does not change any numerical implementation. A plain `pip install mojolearn`
stays NumPy-free.

## Run it

```sh
python -m mojolearn verify --quick     # representative checks, seconds to minutes
python -m mojolearn verify             # every applicable routine lane, base fixture
python -m mojolearn verify --all       # the same scope on all nine fixtures
```

The verifier selects the `identical` mode itself when `MOJOLEARN_NUMERIC_MODE` is unset. A mode you
set explicitly is respected, and a process in `fast` (an Apple GPU mode) is refused (exit 3), because
`fast` makes no identity promise to check.

## Scopes

| Command | What it runs |
|---|---|
| `verify` | Inference and classical training on every applicable lane, base fixture. Includes saved-model inference and, on a GPU install, GPU-against-CPU inference comparisons. |
| `verify --quick` | One representative lane per family from the same scope, base fixture. |
| `verify --all` (alias `--full`) | The routine scope on every fixture, plus batch-size, ragged-batch, gradient and sampler/replay probes where they apply. |
| `verify --neural-training` | Neural training, backward and optimizer fixtures. Not part of the default, quick or all scopes; combine with `--all` for its full fixture set. |
| `verify --coverage` | Lists lanes, CPU availability, batch contracts and reference counts without fitting anything. |
| `verify --cross-check [quick\|default\|all]` | Only the GPU-against-CPU comparisons on your own machine (below). |
| `verify --lanes a,b` | Only the named lanes. |

Add `--json-out PATH` to write the run as an evidence document, or `--json` to print it on stdout:

```sh
python -m mojolearn verify --all --json-out verification.json
python -m mojolearn verify --coverage --json-out coverage.json
```

On a GPU install, `--all` holds the GPU for the whole run, and some lanes take minutes each. On a
shared GPU, run it in pieces with `--lanes`, or start with `--quick`.

## Your GPU against your CPU

```sh
python -m mojolearn verify --cross-check
```

Each lane is fitted once on your GPU. The same fitted model then answers the same held-out query
twice: from the GPU estimator, and from the saved model reloaded through the CPU binding. Both sides
are computed on your machine with one digest implementation, so this check does not depend on the
recorded reference table. Where a lane has one, it also checks batch invariance: the same row keeps
its output bits when its batch neighbors change.

| Tier | Lanes | Fixtures |
|---|---|---|
| `quick` | One eligible lane per family | Base |
| `default` | Up to 24 eligible lanes | Base |
| `all` | Every eligible lane | Every fixture |

On a CPU-only install there is no second device, so the cross-check says so and exits 4. That is
neither a pass nor a failure.

## Reading the result

| Exit | Verdict |
|---|---|
| 0 | `VERIFIED`: every judged part matched; nothing divergent, refused or owed |
| 1 | `MISMATCH`: at least one part differs from its reference |
| 2 | Invalid usage |
| 3 | Refused: the process loaded a mode other than `identical` |
| 4 | `INCOMPLETE` or cannot run: a part was refused, the import failed, or the fixtures differ |
| 5 | Missing reference or incomplete scope: a part is `OWED` |

A part that did not run is never counted as a pass. A lane with no recorded reference reads `OWED`,
not `IDENTICAL`.

## What a pass proves, and what it does not

A `VERIFIED` result says that this build, on this device, reproduced the recorded bits on every part
it ran. Because the references were recorded on other vendors' hardware, a matching part on your
machine joins those columns: your machine and the recorded NVIDIA, AMD, Apple and CPU results give the
same bits there.

It does not:

- re-measure the other vendors;
- cover inputs, shapes or parameters other than the fixtures;
- say anything about speed;
- on a CPU-only install, check GPU training (it checks the CPU reference lanes and the saved models).

Recorded hashes belong to one release. Verify against the hashes shipped with the wheel you installed.

## Show that the verifier can fail

```sh
python -m mojolearn verify --self-test
```

This runs one lane twice through the ordinary comparison: once unchanged, and once with every value of
the input's first column moved up by one ULP. It exits 0 only if the first reads `IDENTICAL` and the
second `DIVERGENT`.

## Compare two machines

Two people on different hardware can check the claim against each other without the project's table:

```sh
# on machine A
python -m mojolearn verify --all --json-out mine.json
# on machine B
python -m mojolearn verify --all --json-out theirs.json
# anywhere: needs no GPU, no bindings and no network
python -m mojolearn verify --compare mine.json theirs.json
```

`--compare` checks every cell hash in the two documents, prints both provenance blocks side by side
and says whether the machines were genuinely different. A cell present in only one document is
`INCOMPARABLE`, never a match. Run the same scope on both machines. To prevent either side from
tailoring its document after seeing the other, use `--commitment` first (see `python -m mojolearn verify --help`).

## Sharing a report

```sh
python -m mojolearn verify --all --json > mojolearn-verify.json
```

The report carries the version and build witness, the numeric mode, the vendor and device, the CPU
model, the platform, Python and NumPy versions, the SHA-256 of every loaded binding and of the
reference table, and every compared part with its reference. Attach it to an issue as is. See
[How do I report a mismatch?](../notes/faq.md).
