# Start here

This is the whole path from a clone to a merged change. It is deliberately
short. **You do not need to read the rest of this repository's documentation
before your first contribution.** The long documents exist so that a claim can
be audited years later, not so that you have to absorb them to fix a bug.

## 1. What you need

One GPU supported by the installed Mojo toolchain and this checkout. There is
no CPU path. Released wheels support a narrower, explicitly packaged set of
architectures; see [the support matrix](../SUPPORT_MATRIX.md) rather than
inferring wheel support from a GPU family name.

You do not need to rent hardware or own a second vendor to contribute. The
three-vendor certificates close cross-vendor identity claims, and running
those legs is a maintainer job.

You also need [pixi](https://pixi.sh). Everything else is in `pixi.toml`.

## 2. Get it running

```sh
git clone https://github.com/mojolearn/mojolearn && cd mojolearn
pixi run probe
```

`probe` is the one command that answers "does this still work". It builds and
runs the correctness suite end to end on your GPU. There is no separate build
step to keep in sync.

If you only want to USE the library rather than work on it, skip all of the
above and `pip install mojolearn`.

## 3. Run one check and read what it prints

Every check in this tree prints the numeric mode it compiled in, the vendor it
ran on, and what it compared. Pick any of them.

```sh
pixi run check-hist
pixi run check-bootstrap
pixi run check-vendor-correctness
```

A check that passes is not evidence on its own. What makes it evidence is that
somebody has shown it can FAIL. That idea is the one thing worth understanding
before you change anything numerical, and section 5 is the short version.

## 4. Good first contributions

In rough order of how self-contained they are.

- **A hardware card.** Run `pixi run probe` on a GPU nobody here has used and
  report what happened. A new column is a real contribution and it costs you
  one command. Failures are more useful than passes.
- **A bug reproduction.** The smallest input that produces the wrong answer,
  with the mode and the vendor named.
- **Documentation that is wrong.** This tree moves fast and sentences go stale.
  Deleting a false sentence is a welcome patch. Do not soften it, delete it.
- **A fixture that separates two spellings.** See section 5.
- **An estimator in the `fast` tier.** Ordinary correctness tests, ordinary
  review. The `identical` tier is a later promotion, not an entry requirement.

## 5. The one rule that is different here

Most projects accept a passing test as evidence. This one does not, for
numerical changes.

If you add a numerical pin, the test for it must first be shown to FAIL when
the pin is removed or spelled the other way. A test that passes under both
spellings has measured nothing, however green it is. In this tree that
deliberate break is called a sabotage arm, and every claim of the form "this is
bit-identical" is backed by one.

So a numerical contribution has two halves. The change, and the demonstration
that its check can tell the difference. If you send the first half only, that
is fine, say so, and a maintainer will work out the second with you.

Everything else is ordinary. Bug fixes, docs, tests, performance work and
tooling follow the same rules as any other repository.

## 6. What to put in the pull request

State what you actually ran. Which GPU, which vendor, which of the three
numeric modes.

Mark any column you did not run `cross-vendor-pending`. Do not infer it, do not
reason about it, and do not leave it blank. An honestly incomplete contribution
merges. An inferred column does not, and it is the single thing most likely to
get a patch sent back.

## 7. Only now, the deep end

Read these when you need them, not before.

| file | what it is for |
|---|---|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | the full contribution rules, provenance and licensing |
| [PORTING_RULES.md](../PORTING_RULES.md) | how upstream algorithms are mirrored, and the rule that assumes our code is the broken one |
| [SUPPORT_MATRIX.md](../SUPPORT_MATRIX.md) | what the three numeric tiers promise, measured rather than argued |
| [IDENTITY_PATHS.md](../IDENTITY_PATHS.md) | which code paths carry a bit-level claim |
| [GOVERNANCE.md](../GOVERNANCE.md) | how decisions are made and how maintainership transfers |
