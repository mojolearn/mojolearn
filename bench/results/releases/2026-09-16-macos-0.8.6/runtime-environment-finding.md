# Loading the Mojo runtime changes the C environment of a macOS process (0.8.6 release evidence)

Recorded on 2026-09-15 on the release Mac (Apple M4, Homebrew CPython 3.10 to 3.14)
during the 0.8.6 macOS wheel verification. Kept here so it can be reported to Modular
later; nobody has been contacted.

## What happens

After `import mojolearn` in a venv, the process's C-level environment (what `environ`
holds, and what a child process inherits) gains three variables that Python's
`os.environ` does not show:

    MOJO_PYTHON_LIBRARY = /opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/Python.framework/Versions/3.14/Python
    PYTHONEXECUTABLE    = /opt/homebrew/bin/python3
    PYTHONPATH          = :

Nothing is removed or changed, `os.environ` still equals the environment before the
import, and `sys.executable` in the parent is unchanged.

## Why it matters

A child started with `subprocess.run([sys.executable, ...])` and no `env=` inherits
the C environment. On Python 3.11 and later, a venv interpreter started with
`PYTHONEXECUTABLE` pointing at the base interpreter leaves its venv (`sys.prefix ==
sys.base_prefix`), so the child cannot import packages installed in the venv. On 3.10
the child stays in the venv.

`MOJO_PYTHON_LIBRARY` names the Homebrew Python 3.14 framework even inside a 3.11
process, which is a separate oddity worth reporting with it.

## Measurements (installed 0.8.6 macOS wheel built at 2f53960ca, sha256 eba69f83)

| probe | 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
|---|---|---|---|---|---|
| parent `sys.executable` changed by the import | no | no | no | no | no |
| child with the inherited environment starts in | venv | base | base | base | base |

On 3.11, from the same installed wheel:

- `/usr/bin/env` run as a child before the import and after it: the after listing adds
  exactly the three variables above.
- a child with the inherited environment starts in the base interpreter; a child with
  `env=dict(os.environ)` starts in the venv.
- `python -m mojolearn identity --check` as a child with the inherited environment exits 1
  with `/opt/homebrew/bin/python3: No module named mojolearn`; with
  `env=dict(os.environ)` it resolves the installed package.

The published 0.8.5 macOS wheel (sha256 matched PyPI) shows the same child behavior on
3.10 (venv) and 3.11 (base).

## Effect on mojolearn and what changed

- `packaging/macos/verify_wheel.sh` ran `identity --check` as a child without `env=`, so
  the 0.8.6 verification failed on 3.11 and 3.12. Fixed in 68906d504 (release/0.8.6
  db9047b9f, main 3f8b5bd69): the child gets `env=dict(os.environ)`.
- The package's own child processes (`_identity._run`, `_verify_all._reexec_identical`,
  `_parallel_pool` workers) already pass environments built from `os.environ`, so a user
  of the installed package is not affected by this path.
- Any user code in the same process that starts a Python child without `env=` after
  importing mojolearn is affected on 3.11 and later.

## To report

The runtime setting `PYTHONEXECUTABLE` and `PYTHONPATH=:` in the host process
environment, and `MOJO_PYTHON_LIBRARY` naming a different Python version than the
running interpreter.
