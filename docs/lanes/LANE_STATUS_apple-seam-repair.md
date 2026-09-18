# LANE STATUS: apple-seam-repair (branch `lane/apple-seam-repair`)

Written for a session with no context. Brief (binding, verbatim):
`~/mojolearn-evidence/relaunch-sep18/BRIEF_apple-seam-repair_original.txt`.
Andrew's directive: Apple MUST compute the contract's round-then-flush (`rtf`)
seam bitwise identically to NVIDIA and AMD. No exception, no narrowing.

## Registered prediction (before any run, 2026-09-18)

`fbr` (Apple's native flush-before-round) and `rtf` agree on every input
except products landing in `[2^-126 - 2^-150, 2^-126)`. No identity card or LM
witness has landed there. So the repair is predicted **INERT on every
existing recorded Apple cell**. Falsifier: any existing Apple cell's hash
moves -> the repair changed something besides the boundary; find out what
before shipping.

Affirmative proof: seam probe on Apple must hash `rtf` = `62a6b5621e27c707`
over 262,144 triples (pre-repair: `fbr` = `f269fc70e5625987`).

## Progress log

- 2026-09-18 session 1 killed ~10 min in (no edits, no runs).
- 2026-09-18 session 2 (this): resumed. Metal slot was held by
  release_runner (pid 92580, release-087-final) at start; queued behind it.
