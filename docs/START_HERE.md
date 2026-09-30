# Start here

This is the short path from a clone to a merged change. You do not need to
read the rest of the documentation before your first contribution.

If you only want to use the library, run `pip install mojolearn` and see the
[README](../README.md).

## 1. What you need

- [pixi](https://pixi.sh). Everything else, including the Mojo toolchain, is
  pinned in `pixi.toml`.
- Optionally, a GPU supported by the Mojo toolchain (Apple, NVIDIA or AMD) to
  build and check GPU paths. Routine checks run on the CPU. Released wheels
  support a narrower, explicitly packaged set of architectures, listed in the
  [support matrix](../SUPPORT_MATRIX.md).

You do not need to rent hardware or own a second vendor. Cross-vendor
certification is run by maintainers.

## 2. Set up the environment

```sh
git clone https://github.com/mojolearn/mojolearn && cd mojolearn
pixi install
sh tools/hooks/install.sh   # commit and push size checks, once per clone
```

The test tools live in a separate pixi environment, used as
`pixi run -e test <task>`.

## 3. Build

Each Python extension is built by its own script under `bindings/`. For
example, `sh bindings/build_gbdt.sh` builds the gradient boosting binding
and `sh bindings/build_gbdt_host.sh` builds its CPU counterpart. Build only
the bindings for the code you are changing.

## 4. Run a check

```sh
pixi run probe
```

`probe` builds and runs the correctness suite end to end on your GPU and is
the quickest answer to "does this still work". Individual checks are pixi
tasks named `check-*`, for example `pixi run check-hist` or
`pixi run check-bootstrap`. Each prints the numeric mode it compiled in, the
device it ran on and what it compared.

For one algorithm on the CPU, with a bounded time budget, use the iteration
runner.

```sh
pixi run -e test test-algo --lane transformer --plan
```

[TEST_RUNTIME.md](TEST_RUNTIME.md) explains its options. The installed-wheel
verifier is described in [VERIFY.md](VERIFY.md).

## 5. Good first contributions

- **A hardware report.** Run `pixi run probe` on a GPU nobody here has used
  and report the result. Failures are as useful as passes.
- **A bug reproduction.** The smallest input that produces the wrong answer,
  with the numeric mode and the device named.
- **A documentation fix.** Delete or correct a sentence that is wrong.
- **A separating fixture.** A test that tells two numerical spellings apart.
- **A new estimator.** It enters in the `identical` mode with ordinary tests.

## 6. The rule that is different here

A passing test is not evidence for a numerical change on its own. A test
counts only if it has been shown to fail when the change is removed or
spelled the other way. A numerical contribution therefore has two halves,
the change and the demonstration that its check can tell the difference. If
you can send only the first half, say so and a maintainer will help with the
second. The full rules are in
[CONTRIBUTING.md](../CONTRIBUTING.md#engineering-rules).

## 7. What to put in the pull request

State what you ran, on which device and in which numeric mode. Mark every
vendor column you did not run `cross-vendor-pending`. An honestly incomplete
contribution merges. An inferred column does not.

## 8. Further reading

| file | what it covers |
|---|---|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | contribution and engineering rules, licensing |
| [SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md) | verified devices and what each numeric mode promises |
| [IDENTITY_PATHS.md](../IDENTITY_PATHS.md) | which code paths carry a bitwise identity claim |
| [VERIFY.md](VERIFY.md) | the verifier and how to read its output |
| [TEST_RUNTIME.md](TEST_RUNTIME.md) | bounded test iteration |
| [ROADMAP.md](../ROADMAP.md) | current direction |
| [GOVERNANCE.md](../GOVERNANCE.md) | how decisions are made |
