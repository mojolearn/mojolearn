# Draft upstream bug report: `DeadArgumentElimination surveyUse failed` at -O2 and above

Status: DRAFT, NOT FILED. Written 2026-09-09 by the dbscan compiler-crash
lane for Andrew to file (or not) against the Mojo repository. Everything
below was measured; nothing is inferred from the pass name.

## Title

`mojo build -O2`/`-O3` aborts with "DeadArgumentElimination surveyUse
failed. UNREACHABLE executed!" on a `while` loop that re-reads `len()` of
a List argument with a runtime Int stride

## Versions

- Mojo 1.0.0 (ed45d567), installed through pixi (`mojo = ">=1.0.0,<2"`,
  `max = ">=26.5.0,<27"` in a locked environment).
- Reproduced on Linux x86-64 (Ubuntu 22.04 container, RunPod NVIDIA L40S,
  driver 595.91.07) on 2026-09-09, and first seen on macOS arm64 (Apple M4)
  on 2026-09-01. The repro is host-only; no GPU code is involved.

## Reproducer

`dbscan/checks/compiler_repro_dead_arg_elim.mojo` in mojolearn; inlined here
so the report is self-contained:

```mojo
def _strided(w: List[Float32], width: Int) -> Float32:
    var acc = Float32(0.0)
    var k = 0
    while k < len(w):
        acc = acc + w[k]
        k += width
    return acc


def main() raises:
    var vals = List[Float32]()
    var tiny = Float32(1.0) / Float32(16777216.0)
    for k in range(256):
        vals.append(Float32(1.0) if k == 0 else tiny)
    var wide = _strided(vals, 128)
    var narrow = _strided(vals, 64)
    if wide == narrow:
        raise Error("same: " + String(wide))
    print(String(wide) + " " + String(narrow))
```

```
$ mojo build -O3 repro.mojo      # aborts (exit 134); -O2 the same
$ mojo build -O1 repro.mojo && ./repro
1.0000151 1.000015
$ mojo build -O0 repro.mojo      # builds
```

## Output (head of the -O3 build log)

```
DeadArgumentElimination surveyUse failed.
UNREACHABLE executed!
PLEASE submit a bug report to https://github.com/modular/modular/issues and include the crash backtrace.
 #0 0x0000622bbc4aa05e (.../bin/mojo+0x822905e)
 #1 0x0000622bbc4a70be (.../bin/mojo+0x82260be)
 #2 0x0000622bbc4aae50 (.../bin/mojo+0x8229e50)
 #3 0x00007badd2b6d520 (/lib/x86_64-linux-gnu/libc.so.6+0x42520)
 #4 0x00007badd2bc19fc pthread_kill (/lib/x86_64-linux-gnu/libc.so.6+0x969fc)
 #5 0x00007badd2b6d476 gsignal (/lib/x86_64-linux-gnu/libc.so.6+0x42476)
 #6 0x00007badd2b537f3 abort (/lib/x86_64-linux-gnu/libc.so.6+0x287f3)
 #7 0x0000622bbc46b1a4 (.../bin/mojo+0x81ea1a4)
 #8 0x0000622bb84767e1 (.../bin/mojo+0x41f57e1)
 #9 0x0000622bbbb9875b (.../bin/mojo+0x791775b)
#10 0x0000622bbbb9b340 (.../bin/mojo+0x791a340)
#11 0x0000622bb894e198 (.../bin/mojo+0x46cd198)
#12 0x0000622bb82b0023 (.../bin/mojo+0x402f023)
...
```

(The third line is the standard LLVM crash banner; frames are unsymbolized
in the shipped binary. Full log: `bench/results/` is not used for this;
the lane kept it on the pod only. Re-run the two-line command above to
regenerate it.)

## What is and is not needed to trigger it

Measured on the same box, one variable per file, all at -O3
(`docs/lanes/HANDOFF_dbscan_crash.md` has the full ladder):

| variant of the loop above | result |
|---|---|
| `while k < len(w): ...; k += width` (width a runtime `Int` argument) | ASSERTS |
| `var n = len(w)` hoisted, `while k < n: ...; k += width` | builds |
| `for k in range(0, len(w), width)` | builds |
| `while k < len(w): ...; k += 1` (unit stride) | builds |
| `while k < len(w): ...; k += Int(w[0]) + 63` (stride not an argument) | builds |
| `def _strided[width: Int](w)` (comptime stride) | builds |
| `Int` bound argument instead of `len(w)`, runtime stride | builds |
| callee marked `@always_inline` | builds |

Things that do NOT change the outcome: one call site instead of two; a
runtime-derived width instead of literals; `@no_inline`; an owned (`var`)
List argument; `List[Int]` instead of `List[Float32]`; a second List
indexed through the first; an inner `List` of partials or none.

So the trigger is the combination of (a) `len()` of a borrowed `List`
argument re-evaluated in the `while` condition and (b) the induction step
being a runtime argument of the same function. Either alone is fine.

## Impact

A 2,400-line check file in mojolearn could not be built at -O2 or above for
eight days; the workaround was `-O1` on one pixi task and two shell gates.
The one-line hoist above is now in the source and the -O3 build's output is
byte-identical to the -O1 build's.
