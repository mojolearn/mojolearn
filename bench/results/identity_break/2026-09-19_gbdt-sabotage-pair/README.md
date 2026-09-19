# The gbdt five: one clean arm and one sabotage arm, same commit

`Sabotage not seen to move a build: 5` was the last non-par gap in
docs/VERIFICATION_MATRIX.md. It was never that the sabotage failed to
move these lanes; it was that no COMMITTED pair said so.

Both columns are `a2c20c8c0`, `--repeats 2`, base fixture, the CPU
column (`--require-cpu`), seven gbdt lanes. The arm is the generic
gbdt host-binding sabotage build; the clean arm is the same build set
without it, run from the same package.

Every one of the seven moves on five parts -- infer, model, batch,
hashes, reload. `batch` is the one that matters for the second count:
these lanes declare a real batch part, and until now nothing showed a
sabotage disturbing it.

WHY THE PAIR WAS RE-RUN RATHER THAN COMMITTED AS FOUND. An earlier
set of six arms existed but its clean column was recorded at
bca319df2 while the arms were at 900038b24, five commits later. A pair
whose commits differ is kept and MARKED by `sabotage_moves`, because
two commits can differ for reasons that are not the sabotage -- so the
move would not have been attributable to the arm. Re-running both
halves at one commit costs ten minutes of CPU and removes the doubt.
The original six-arm evidence is kept outside the repo at
~/mojolearn-evidence/gbdt-sabotage-sep19; it is what showed that the
generic arm alone moves all five lanes, which is why one arm suffices
here.
